/**
 * Validates the GPU-resident autograd engine against the trusted CPU `Tensor`,
 * then trains an MLP entirely on the GPU and benchmarks it.
 *
 *   deno run --unstable-webgpu --unstable-sloppy-imports --allow-read \
 *     scripts/gpu-train.ts
 *
 * Strategy: build the *same* computation with the same initial weights on both
 * engines. If every forward value and every parameter gradient matches to f32
 * tolerance, the GPU backward passes are correct. The script exits non-zero on
 * any mismatch, so it is a test, not a demo.
 */

import { Tensor } from "../src/math/tensor.ts";
import { mulberry32, gaussian } from "../src/math/random.ts";
import { GpuEngine, GpuTensor } from "../src/infer/webgpu/autograd.ts";

function randn(n: number, seed: number, std = 1): Float32Array {
  const rng = mulberry32(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = gaussian(rng) * std;
  return out;
}

function maxRelErr(a: Float32Array, b: ArrayLike<number>): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    const denom = Math.max(1e-3, Math.abs(a[i]), Math.abs(b[i]));
    worst = Math.max(worst, Math.abs(a[i] - b[i]) / denom);
  }
  return worst;
}

const TOL = 2e-3;
let failed = false;
function check(label: string, gpu: Float32Array, cpu: ArrayLike<number>): void {
  const err = maxRelErr(gpu, cpu);
  const ok = err <= TOL;
  console.log(`  ${ok ? "ok  " : "FAIL"} ${label.padEnd(22)} max rel err ${err.toExponential(2)}`);
  if (!ok) failed = true;
}

async function main(): Promise<void> {
  const e = await GpuEngine.create();
  console.log("GPU-resident autograd — validation vs CPU engine\n");

  // A two-layer MLP: y = relu(x @ W1 + b1) @ W2 + b2 ; loss = mean((y - target)^2)
  const B = 32;
  const din = 24;
  const hid = 40;
  const dout = 12;

  const xData = randn(B * din, 1);
  const tData = randn(B * dout, 2);
  const w1Data = randn(din * hid, 3, 0.2);
  const b1Data = new Float32Array(hid);
  const w2Data = randn(hid * dout, 4, 0.2);
  const b2Data = new Float32Array(dout);

  // ---- CPU reference -------------------------------------------------------
  const buildCpu = () => {
    const x = new Tensor(xData.slice(), [B, din]);
    const target = new Tensor(tData.slice(), [B, dout]);
    const w1 = new Tensor(w1Data.slice(), [din, hid], true);
    const b1 = new Tensor(b1Data.slice(), [hid], true);
    const w2 = new Tensor(w2Data.slice(), [hid, dout], true);
    const b2 = new Tensor(b2Data.slice(), [dout], true);
    const h = x.matmul(w1).add(b1).relu();
    const y = h.matmul(w2).add(b2);
    const d = y.sub(target);
    const loss = d.mul(d).mean();
    return { loss, params: { w1, b1, w2, b2 }, y };
  };
  const cpu = buildCpu();
  cpu.loss.backward();

  // ---- GPU -----------------------------------------------------------------
  const x = e.tensor(xData, [B, din]);
  const target = e.tensor(tData, [B, dout]);
  const w1 = e.tensor(w1Data, [din, hid], true);
  const b1 = e.tensor(b1Data, [hid], true);
  const w2 = e.tensor(w2Data, [hid, dout], true);
  const b2 = e.tensor(b2Data, [dout], true);
  const forward = (): GpuTensor => {
    const h = x.matmul(w1).add(b1).relu();
    const y = h.matmul(w2).add(b2);
    const d = y.sub(target);
    return d.mul(d).mean();
  };
  const loss = forward();
  loss.backward();

  console.log("Forward + backward (single step):");
  check("loss", await e.read(loss.buffer, 1), [cpu.loss.item()]);
  check("dW1", await e.read(w1.grad!, w1.size), cpu.params.w1.grad!);
  check("db1", await e.read(b1.grad!, b1.size), cpu.params.b1.grad!);
  check("dW2", await e.read(w2.grad!, w2.size), cpu.params.w2.grad!);
  check("db2", await e.read(b2.grad!, b2.size), cpu.params.b2.grad!);

  // ---- Training: both engines take the same SGD steps ----------------------
  console.log("\nTraining 100 SGD steps (GPU) vs CPU:");
  const lr = 0.05;
  const gpuParams = [w1, b1, w2, b2];

  // CPU training loop.
  const cpuW = {
    w1: new Tensor(w1Data.slice(), [din, hid], true),
    b1: new Tensor(b1Data.slice(), [hid], true),
    w2: new Tensor(w2Data.slice(), [hid, dout], true),
    b2: new Tensor(b2Data.slice(), [dout], true),
  };
  const xC = new Tensor(xData.slice(), [B, din]);
  const tC = new Tensor(tData.slice(), [B, dout]);
  let cpuLoss = 0;
  for (let step = 0; step < 100; step++) {
    for (const p of Object.values(cpuW)) p.zeroGrad();
    const h = xC.matmul(cpuW.w1).add(cpuW.b1).relu();
    const y = h.matmul(cpuW.w2).add(cpuW.b2);
    const d = y.sub(tC);
    const l = d.mul(d).mean();
    l.backward();
    for (const p of Object.values(cpuW)) for (let i = 0; i < p.size; i++) p.data[i] -= lr * p.grad![i];
    cpuLoss = l.item();
  }

  // GPU training loop (reset params to the same init first).
  e.device.queue.writeBuffer(w1.buffer, 0, w1Data);
  e.device.queue.writeBuffer(b1.buffer, 0, b1Data);
  e.device.queue.writeBuffer(w2.buffer, 0, w2Data);
  e.device.queue.writeBuffer(b2.buffer, 0, b2Data);
  let gpuLoss = 0;
  for (let step = 0; step < 100; step++) {
    for (const p of gpuParams) p.zeroGrad();
    const l = forward();
    l.backward();
    for (const p of gpuParams) e.sgd(p.grad!, p.buffer, p.size, lr);
    if (step === 99) gpuLoss = (await e.read(l.buffer, 1))[0];
  }
  console.log(`  CPU final loss ${cpuLoss.toFixed(6)}`);
  console.log(`  GPU final loss ${gpuLoss.toFixed(6)}`);
  check("final loss match", new Float32Array([gpuLoss]), [cpuLoss]);

  if (failed) {
    console.error("\nVALIDATION FAILED");
    Deno.exit(1);
  }
  console.log("\nAll GPU gradients match the CPU engine within tolerance.");
}

main().catch((err) => {
  console.error(err);
  Deno.exit(1);
});
