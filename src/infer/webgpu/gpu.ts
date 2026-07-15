/**
 * WebGPU compute core for Turt.
 *
 * This module is deliberately framework-agnostic: it depends only on the
 * standard WebGPU API and operates on `Float32Array` + shapes, never on Turt's
 * `Tensor`. That keeps it runnable in any WebGPU host — Deno on Apple Metal
 * (how it is validated here), a Node WebGPU binding, a browser, or a cloud
 * NVIDIA GPU — and keeps the numerical core testable without the autograd
 * machinery around it. A thin `GpuBackend` adapter (see gpu-backend.ts) bridges
 * it to `Tensor` for the inference engine.
 *
 * The pipeline is compiled once and cached; each call allocates the buffers it
 * needs, dispatches, and reads back. Keeping buffers resident across ops is the
 * next optimization (see docs/GPU.md) and is what turns this from a fast kernel
 * into a fast training loop.
 */

import { MATMUL_WGSL, TILE } from "./kernels.js";

/** Minimal shape of the WebGPU device, so this file needs no DOM lib types. */
interface GpuLike {
  gpu: {
    requestAdapter(options?: unknown): Promise<GpuAdapterLike | null>;
  };
}
interface GpuAdapterLike {
  requestDevice(): Promise<GPUDevice>;
  info?: { vendor?: string; device?: string; architecture?: string };
}

/** Resolves a `GPUDevice` from the ambient `navigator.gpu`, or returns null. */
export async function requestGpuDevice(): Promise<GPUDevice | null> {
  const nav = (globalThis as unknown as { navigator?: GpuLike }).navigator;
  if (!nav?.gpu) return null;
  const adapter = await nav.gpu.requestAdapter();
  if (!adapter) return null;
  return adapter.requestDevice();
}

interface MatShape {
  batch: number;
  m: number;
  k: number;
  n: number;
}

export class GpuCompute {
  private readonly pipeline: GPUComputePipeline;

  private constructor(readonly device: GPUDevice) {
    const module = device.createShaderModule({ code: MATMUL_WGSL });
    this.pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });
  }

  /** Creates a compute context, acquiring a device if one is not supplied. */
  static async create(device?: GPUDevice): Promise<GpuCompute> {
    const dev = device ?? (await requestGpuDevice());
    if (!dev) throw new Error("No WebGPU device available in this environment");
    return new GpuCompute(dev);
  }

  /** 2-D matrix multiply: [m,k] @ [k,n] -> [m,n]. */
  async matmul(a: Float32Array, m: number, k: number, b: Float32Array, n: number): Promise<Float32Array> {
    return this.run(a, b, { batch: 1, m, k, n });
  }

  /** Batched matrix multiply: [g,m,k] @ [g,k,n] -> [g,m,n]. */
  async bmm(
    a: Float32Array,
    batch: number,
    m: number,
    k: number,
    b: Float32Array,
    n: number,
  ): Promise<Float32Array> {
    return this.run(a, b, { batch, m, k, n });
  }

  private async run(a: Float32Array, b: Float32Array, shape: MatShape): Promise<Float32Array> {
    const { batch, m, k, n } = shape;
    if (a.length !== batch * m * k) throw new Error(`A has ${a.length} elems, expected ${batch * m * k}`);
    if (b.length !== batch * k * n) throw new Error(`B has ${b.length} elems, expected ${batch * k * n}`);

    const device = this.device;
    const outElems = batch * m * n;

    const bufA = device.createBuffer({
      size: a.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bufB = device.createBuffer({
      size: b.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const bufC = device.createBuffer({
      size: outElems * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const bufDims = device.createBuffer({
      size: 16, // 4 x u32, padded to 16 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(bufA, 0, a);
    device.queue.writeBuffer(bufB, 0, b);
    device.queue.writeBuffer(bufDims, 0, new Uint32Array([m, k, n, batch]));

    const bindGroup = device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufA } },
        { binding: 1, resource: { buffer: bufB } },
        { binding: 2, resource: { buffer: bufC } },
        { binding: 3, resource: { buffer: bufDims } },
      ],
    });

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(n / TILE), Math.ceil(m / TILE), batch);
    pass.end();

    const readback = device.createBuffer({
      size: outElems * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    encoder.copyBufferToBuffer(bufC, 0, readback, 0, outElems * 4);
    device.queue.submit([encoder.finish()]);

    await readback.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readback.getMappedRange().slice(0));
    readback.unmap();

    for (const buf of [bufA, bufB, bufC, bufDims, readback]) buf.destroy();
    return result;
  }
}
