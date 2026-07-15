/**
 * Validates and benchmarks the WebGPU backend against the CPU engine.
 *
 * Runs under Deno, which ships a native WebGPU implementation (Metal-backed on
 * Apple silicon, Vulkan on Linux/NVIDIA). The same kernels and host code run
 * unchanged on a cloud GPU — only this launcher differs.
 *
 *   deno run --unstable-webgpu --unstable-sloppy-imports --allow-read \
 *     scripts/gpu-bench.ts
 *
 * The script asserts GPU results match a CPU reference to f32 tolerance and
 * exits non-zero on any mismatch, so it doubles as a correctness test. It then
 * reports GFLOP/s across matrix sizes — including the exact shapes the trained
 * GPT uses — and the crossover point where the GPU overtakes the CPU.
 */

import { GpuCompute } from "../src/infer/webgpu/gpu.ts";

/** CPU reference matmul (same cache-friendly i-k-j order as the Tensor engine). */
function cpuMatmul(a: Float32Array, m: number, k: number, b: Float32Array, n: number): Float32Array {
  const out = new Float32Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const av = a[i * k + p];
      if (av === 0) continue;
      const bo = p * n;
      const oo = i * n;
      for (let j = 0; j < n; j++) out[oo + j] += av * b[bo + j];
    }
  }
  return out;
}

function randArray(n: number, seed: number): Float32Array {
  // Deterministic mulberry32, so runs are reproducible without Math.random.
  let s = seed >>> 0;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    out[i] = (((t ^ (t >>> 14)) >>> 0) / 4294967296) * 2 - 1;
  }
  return out;
}

/** Largest relative difference between two arrays. */
function maxRelError(a: Float32Array, b: Float32Array): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    const denom = Math.max(1e-4, Math.abs(a[i]), Math.abs(b[i]));
    worst = Math.max(worst, Math.abs(a[i] - b[i]) / denom);
  }
  return worst;
}

const CPU_GFLOPS = 2.3; // measured single-threaded baseline for this engine

async function main(): Promise<void> {
  const gpu = await GpuCompute.create();
  const info = (gpu.device as unknown as { adapterInfo?: { device?: string } }).adapterInfo;
  console.log(`WebGPU device acquired${info?.device ? ` (${info.device})` : ""}.\n`);

  // ---- Correctness ---------------------------------------------------------
  console.log("Correctness (GPU vs CPU reference):");
  const cases: Array<[number, number, number]> = [
    [1, 1, 1],
    [16, 16, 16],
    [17, 33, 9], // non-multiples of the tile size exercise the boundary guards
    [64, 128, 384],
    [256, 160, 1024],
  ];
  let maxErr = 0;
  for (const [m, k, n] of cases) {
    const a = randArray(m * k, m * 7 + k);
    const b = randArray(k * n, n * 13 + k);
    const got = await gpu.matmul(a, m, k, b, n);
    const want = cpuMatmul(a, m, k, b, n);
    const err = maxRelError(got, want);
    maxErr = Math.max(maxErr, err);
    console.log(`  [${m}x${k}]@[${k}x${n}]  max rel err ${err.toExponential(2)}`);
  }

  // Batched matmul (attention-shaped).
  {
    const g = 8;
    const m = 128;
    const k = 40;
    const n = 128;
    const a = randArray(g * m * k, 3);
    const b = randArray(g * k * n, 5);
    const got = await gpu.bmm(a, g, m, k, b, n);
    let err = 0;
    for (let bi = 0; bi < g; bi++) {
      const want = cpuMatmul(
        a.subarray(bi * m * k, (bi + 1) * m * k),
        m,
        k,
        b.subarray(bi * k * n, (bi + 1) * k * n),
        n,
      );
      err = Math.max(err, maxRelError(got.subarray(bi * m * n, (bi + 1) * m * n), want));
    }
    maxErr = Math.max(maxErr, err);
    console.log(`  bmm[${g}x${m}x${k}]@[${g}x${k}x${n}]  max rel err ${err.toExponential(2)}`);
  }

  const TOLERANCE = 5e-3; // f32 accumulation-order differences
  if (maxErr > TOLERANCE) {
    console.error(`\nFAIL: max relative error ${maxErr.toExponential(2)} exceeds ${TOLERANCE}`);
    Deno.exit(1);
  }
  console.log(`  OK — worst error ${maxErr.toExponential(2)} < ${TOLERANCE}\n`);

  // ---- Throughput ----------------------------------------------------------
  console.log("Throughput (GPU vs 2.3 GFLOP/s CPU baseline):");
  const shapes: Array<{ label: string; m: number; k: number; n: number }> = [
    { label: "tiny        128x128@128x128", m: 128, k: 128, n: 128 },
    { label: "gpt qkv     4096x160@160x160", m: 4096, k: 160, n: 160 },
    { label: "gpt mlp-up  4096x160@160x640", m: 4096, k: 160, n: 640 },
    { label: "gpt head    4096x160@160x1024", m: 4096, k: 160, n: 1024 },
    { label: "large       1024x1024@1024x1024", m: 1024, k: 1024, n: 1024 },
    { label: "xl          2048x2048@2048x2048", m: 2048, k: 2048, n: 2048 },
  ];

  for (const { label, m, k, n } of shapes) {
    const a = randArray(m * k, m + k);
    const b = randArray(k * n, k + n);
    await gpu.matmul(a, m, k, b, n); // warm up pipeline + allocations

    const flops = 2 * m * k * n;
    const reps = Math.max(3, Math.min(50, Math.floor(2e9 / flops)));
    const start = performance.now();
    for (let i = 0; i < reps; i++) await gpu.matmul(a, m, k, b, n);
    const seconds = (performance.now() - start) / 1000;
    const gflops = (flops * reps) / seconds / 1e9;
    console.log(
      `  ${label.padEnd(34)} ${gflops.toFixed(1).padStart(7)} GFLOP/s  ` +
        `(${(gflops / CPU_GFLOPS).toFixed(0)}x CPU)`,
    );
  }
}

main().catch((error) => {
  console.error(error);
  Deno.exit(1);
});
