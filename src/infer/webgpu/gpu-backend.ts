import { Tensor } from "../../math/tensor.js";
import type { Backend } from "../backend.js";
import { GpuCompute, requestGpuDevice } from "./gpu.js";

/**
 * WebGPU-backed `Backend`, bridging the framework-agnostic `GpuCompute` core to
 * Turt's `Tensor`.
 *
 * This is an *inference* backend: `matmul` returns a detached tensor with no
 * autograd context, which is all the inference engine needs. GPU-accelerated
 * *training* is a larger change — it requires tensors that stay resident in GPU
 * memory across ops so gradients never round-trip to the CPU (see docs/GPU.md).
 * The `Backend` seam is deliberately the same one the CPU backend implements,
 * so callers dispatch identically regardless of device.
 *
 * `isAvailable()` probes for a device without throwing, so code can prefer the
 * GPU and fall back to `CpuBackend` when WebGPU is absent (e.g. plain Node).
 */
export class GpuBackend implements Backend {
  readonly name = "webgpu";
  private compute: GpuCompute | null = null;

  private constructor(compute: GpuCompute) {
    this.compute = compute;
  }

  /** Returns true if a WebGPU device can be acquired in this runtime. */
  static async isAvailable(): Promise<boolean> {
    try {
      return (await requestGpuDevice()) !== null;
    } catch {
      return false;
    }
  }

  /** Creates a backend, or throws if no device is available. */
  static async create(device?: GPUDevice): Promise<GpuBackend> {
    return new GpuBackend(await GpuCompute.create(device));
  }

  isAvailable(): boolean {
    return this.compute !== null;
  }

  async matmul(a: Tensor, b: Tensor): Promise<Tensor> {
    if (!this.compute) throw new Error("GpuBackend is not initialized");
    if (a.ndim !== 2 || b.ndim !== 2) {
      throw new Error(`GpuBackend.matmul requires 2-D tensors, got ${a.ndim}-D and ${b.ndim}-D`);
    }
    const [m, k] = a.shape;
    const [k2, n] = b.shape;
    if (k !== k2) throw new Error(`matmul shape mismatch: [${a.shape}] @ [${b.shape}]`);
    const out = await this.compute.matmul(a.data, m, k, b.data, n);
    return new Tensor(out, [m, n]);
  }
}
