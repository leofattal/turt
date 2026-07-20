/**
 * Validates the GPU-resident GPT against the CPU model, then benchmarks it.
 *
 *   deno run --unstable-webgpu --unstable-sloppy-imports --allow-read \
 *     scripts/gpu-gpt-validate.ts
 *
 * Validation loads identical weights into both models, runs one forward+backward
 * on the same batch, and compares the loss and *every* parameter gradient to f32
 * tolerance. Then it benchmarks a full resident training step (forward, backward,
 * AdamW) at a realistic size and reports tokens/sec vs the CPU model.
 */

import { GPT, type Arch, type GPTConfig } from "../src/models/gpt.ts";
import { GpuGPT } from "../src/models/gpu-gpt.ts";
import { Tensor } from "../src/math/tensor.ts";
import { mulberry32 } from "../src/math/random.ts";
import { GpuEngine, GpuTensor } from "../src/infer/webgpu/autograd.ts";

function maxRelErr(a: Float32Array, b: ArrayLike<number>): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    const denom = Math.max(1e-3, Math.abs(a[i]), Math.abs(b[i]));
    worst = Math.max(worst, Math.abs(a[i] - b[i]) / denom);
  }
  return worst;
}

/** Copies CPU parameter values into the GPU model (same order), checking shapes. */
function loadWeights(engine: GpuEngine, cpu: Tensor[], gpu: GpuTensor[]): void {
  if (cpu.length !== gpu.length) throw new Error(`param count mismatch: cpu ${cpu.length}, gpu ${gpu.length}`);
  for (let i = 0; i < cpu.length; i++) {
    if (cpu[i].size !== gpu[i].size) throw new Error(`param ${i} size mismatch`);
    engine.device.queue.writeBuffer(gpu[i].buffer, 0, cpu[i].data);
  }
}

/** Parameter names in `parameters()` order, so a mismatch names the tensor. */
function paramNames(config: GPTConfig): string[] {
  const modern = (config.arch ?? "gpt2") === "modern";
  const names = ["wte"];
  if (!modern) names.push("wpe");
  const perBlock = modern
    ? ["ln1g", "qW", "kW", "vW", "projW", "ln2g", "gateW", "upW", "downW"]
    : ["ln1g", "ln1b", "qW", "qb", "kW", "kb", "vW", "vb", "projW", "projb",
       "ln2g", "ln2b", "fcW", "fcb", "mpW", "mpb"];
  for (let L = 0; L < config.nLayer; L++) names.push(...perBlock.map((s) => `blk${L}.${s}`));
  names.push("lnfg");
  if (!modern) names.push("lnfb");
  return names;
}

async function validate(engine: GpuEngine, arch: Arch): Promise<boolean> {
  console.log(`Validation — GPU GPT vs CPU GPT, arch "${arch}" (identical weights):\n`);
  const config: GPTConfig = { vocabSize: 48, blockSize: 16, nLayer: 2, nHead: 2, nEmbd: 32, arch };
  const b = 2, t = 8;

  const cpuModel = new GPT(config, mulberry32(7));
  const gpuModel = new GpuGPT(engine, config);
  loadWeights(engine, cpuModel.parameters(), gpuModel.parameters());

  const rng = mulberry32(99);
  const tokens = Uint32Array.from({ length: b * t }, () => Math.floor(rng() * config.vocabSize));
  const targets = Uint32Array.from({ length: b * t }, () => Math.floor(rng() * config.vocabSize));

  const cpuLoss = cpuModel.loss(new Tensor(Float32Array.from(tokens), [b, t]), Array.from(targets));
  cpuLoss.backward();
  const gpuLoss = gpuModel.loss(tokens, targets, b, t);
  gpuLoss.backward();

  const lossErr = maxRelErr(await engine.read(gpuLoss.buffer, 1), [cpuLoss.item()]);
  console.log(`  loss   cpu ${cpuLoss.item().toFixed(6)}  gpu ${(await engine.read(gpuLoss.buffer, 1))[0].toFixed(6)}  (rel err ${lossErr.toExponential(2)})`);

  const cpuP = cpuModel.parameters();
  const gpuP = gpuModel.parameters();
  let worst = lossErr;
  let worstName = "loss";
  const names = paramNames(config);
  for (let i = 0; i < cpuP.length; i++) {
    if (!cpuP[i].grad || !gpuP[i].grad) {
      console.log(`  FAIL ${names[i]} missing grad`);
      return false;
    }
    const err = maxRelErr(await engine.read(gpuP[i].grad!, gpuP[i].size), cpuP[i].grad!);
    if (err > worst) { worst = err; worstName = names[i]; }
  }
  const TOL = 5e-3;
  console.log(`  gradients: ${cpuP.length} params compared, worst = ${worst.toExponential(2)} (${worstName})`);
  const ok = worst <= TOL && Number.isFinite(worst);
  console.log(ok ? "\n  PASS — every gradient matches the CPU model.\n" : "\n  FAIL — gradient mismatch.\n");
  return ok;
}

async function benchmark(engine: GpuEngine, arch: Arch, withCpuReference: boolean): Promise<number> {
  const config: GPTConfig = { vocabSize: 1024, blockSize: 128, nLayer: 6, nHead: 4, nEmbd: 160, arch };
  const b = 16, t = config.blockSize;
  const tokensPerStep = b * t;

  const model = new GpuGPT(engine, config);
  const params = model.parameters();
  console.log(`  arch "${arch}": ${(model.numParams() / 1e6).toFixed(2)}M params, batch ${b} x ${t} = ${tokensPerStep} tokens/step`);

  // AdamW state, resident.
  const mBuf = params.map((p) => engine.buffer(p.size));
  const vBuf = params.map((p) => engine.buffer(p.size));
  const rng = mulberry32(1);
  const makeBatch = (): [Uint32Array, Uint32Array] => [
    Uint32Array.from({ length: tokensPerStep }, () => Math.floor(rng() * config.vocabSize)),
    Uint32Array.from({ length: tokensPerStep }, () => Math.floor(rng() * config.vocabSize)),
  ];

  let step = 0;
  const doStep = async (): Promise<number> => {
    for (const p of params) p.zeroGrad();
    const [tok, tgt] = makeBatch();
    const loss = model.loss(tok, tgt, b, t);
    loss.backward();
    step++;
    const bc1 = 1 - Math.pow(0.9, step);
    const bc2 = 1 - Math.pow(0.999, step);
    for (let i = 0; i < params.length; i++) {
      engine.adamw(params[i].grad!, params[i].buffer, mBuf[i], vBuf[i], params[i].size, 3e-4, 0.9, 0.999, 1e-8, 0.1, bc1, bc2);
    }
    return (await engine.read(loss.buffer, 1))[0]; // sync
  };

  await doStep(); // warmup (pipeline compile + allocations)
  const REPS = 8;
  const start = performance.now();
  let last = 0;
  for (let i = 0; i < REPS; i++) last = await doStep();
  const secs = (performance.now() - start) / 1000;
  const tokPerSec = (REPS * tokensPerStep) / secs;
  console.log(`    GPU: ${tokPerSec.toFixed(0)} tok/s  (${(secs / REPS).toFixed(2)}s/step, last loss ${last.toFixed(3)})`);

  if (withCpuReference) {
    // CPU single-thread reference on the same config (one step; it is slow).
    const cpu = new GPT(config, mulberry32(1));
    const [tok, tgt] = makeBatch();
    const idx = new Tensor(Float32Array.from(tok), [b, t]);
    const cstart = performance.now();
    const cl = cpu.loss(idx, Array.from(tgt));
    cl.backward();
    const csecs = (performance.now() - cstart) / 1000;
    const cpuTokPerSec = tokensPerStep / csecs;
    console.log(`    CPU: ${cpuTokPerSec.toFixed(0)} tok/s  (${csecs.toFixed(2)}s/step, single-thread)`);
    console.log(`    Resident GPU training is ${(tokPerSec / cpuTokPerSec).toFixed(1)}x the single-thread CPU trainer.`);
  }
  return tokPerSec;
}

async function main(): Promise<void> {
  const engine = await GpuEngine.create();
  for (const arch of ["gpt2", "modern"] as Arch[]) {
    if (!(await validate(engine, arch))) Deno.exit(1);
  }

  console.log("Benchmark — full resident training step (forward+backward+AdamW):\n");
  const legacy = await benchmark(engine, "gpt2", true);
  console.log();
  const modern = await benchmark(engine, "modern", false);
  // This engine is dispatch-bound, so the modern block's lower op count shows up
  // directly as throughput at matched parameter count — see docs/SCALING.md.
  console.log(`\n  "modern" runs at ${(modern / legacy).toFixed(2)}x the "gpt2" throughput at matched params.`);
}

main().catch((err) => {
  console.error(err);
  Deno.exit(1);
});
