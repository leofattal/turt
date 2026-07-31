/**
 * Pretrains a GPT from scratch, entirely on the GPU.
 *
 *   deno run --unstable-webgpu --unstable-sloppy-imports --allow-read --allow-write \
 *     scripts/gpu-pretrain.ts [--layers 8] [--heads 8] [--embd 512] [--block 256] \
 *     [--batch 16] [--steps 20000] [--lr 6e-4] [--arch modern] \
 *     [--data data] [--out gpt-big.bin] [--resume gpt-big.bin]
 *
 * This is the GPU counterpart to scripts/train-gpt.ts (the 8-worker CPU trainer).
 * The whole forward/backward/AdamW step stays in GPU memory — only token ids go
 * in and the scalar loss comes out — so it runs ~50x faster than the CPU pool and
 * unlocks models an order of magnitude larger than the ~2M-param CPU ceiling.
 *
 * The model size comes from flags rather than the data (unlike finetune-chat.ts,
 * which inherits a checkpoint's config): pretraining is where the architecture is
 * chosen. Vocab and block size are validated against the tokenizer/data. Weights
 * are checkpointed in the portable CPU format (checkpoints/<out>) so generate.ts,
 * the chat CLI, and finetune-chat.ts all consume them unchanged.
 *
 * `--arch` picks the transformer shape: "modern" (default) is RMSNorm + SwiGLU +
 * RoPE with no biases, which reaches a lower loss than the original "gpt2" shape
 * at the same parameter count. `--arch gpt2` reproduces the old runs.
 * Resuming ignores the flag and uses the checkpoint's own architecture.
 */

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { GPT, type Arch, type GPTConfig } from "../src/models/gpt.ts";
import { GpuGPT } from "../src/models/gpu-gpt.ts";
import { GpuEngine, type GpuTensor } from "../src/infer/webgpu/autograd.ts";
import { mulberry32 } from "../src/math/random.ts";
import { BPETokenizer } from "../src/tokenizer/bpe.ts";
import { loadCheckpoint, readCheckpointMeta, saveCheckpoint } from "./checkpoint.ts";

const ROOT = join(import.meta.dirname, "..");
const CKPT_DIR = join(ROOT, "checkpoints");

const numFlag = (name: string, fallback: number): number => {
  const i = Deno.args.indexOf(`--${name}`);
  return i >= 0 && Deno.args[i + 1] ? Number(Deno.args[i + 1]) : fallback;
};
const strFlag = (name: string, fallback: string): string => {
  const i = Deno.args.indexOf(`--${name}`);
  return i >= 0 && Deno.args[i + 1] ? Deno.args[i + 1] : fallback;
};

// --data selects the corpus directory (default "data"), so the big C4 corpus
// from prepare-data-big.ts (data-big/) can coexist with the small one.
const DATA_DIR = join(ROOT, strFlag("data", "data"));

async function loadTokens(name: string): Promise<Uint16Array> {
  const buf = await readFile(join(DATA_DIR, name));
  return new Uint16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2);
}

/** Draws a random contiguous [batch, time] window and its next-token targets. */
function sampleBatch(tokens: Uint16Array, b: number, t: number, rng: () => number): [Uint32Array, Uint32Array] {
  const inp = new Uint32Array(b * t);
  const tgt = new Uint32Array(b * t);
  for (let i = 0; i < b; i++) {
    const s = Math.floor(rng() * (tokens.length - t - 1));
    for (let j = 0; j < t; j++) {
      inp[i * t + j] = tokens[s + j];
      tgt[i * t + j] = tokens[s + j + 1];
    }
  }
  return [inp, tgt];
}

function cpuToGpu(engine: GpuEngine, cpu: GPT, gpu: GpuGPT): void {
  const cp = cpu.parameters();
  const gp = gpu.parameters();
  for (let i = 0; i < cp.length; i++) engine.device.queue.writeBuffer(gp[i].buffer, 0, cp[i].data);
}

async function gpuToCpu(engine: GpuEngine, gpu: GpuGPT, cpu: GPT): Promise<void> {
  const cp = cpu.parameters();
  const gp = gpu.parameters();
  for (let i = 0; i < cp.length; i++) cp[i].data.set(await engine.read(gp[i].buffer, gp[i].size));
}

function fmtDuration(secs: number): string {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  return h > 0 ? `${h}h${m}m` : `${m}m`;
}

async function main(): Promise<void> {
  const meta = JSON.parse(await readFile(join(DATA_DIR, "meta.json"), "utf8")) as { vocabSize: number };

  const arch = strFlag("arch", "modern");
  if (arch !== "modern" && arch !== "gpt2") {
    throw new Error(`--arch must be "modern" or "gpt2", got "${arch}"`);
  }
  const config: GPTConfig = {
    vocabSize: meta.vocabSize,
    blockSize: numFlag("block", 256),
    nLayer: numFlag("layers", 8),
    nHead: numFlag("heads", 8),
    nEmbd: numFlag("embd", 512),
    arch: arch as Arch,
  };
  const b = numFlag("batch", 16);
  const t = config.blockSize;
  const steps = numFlag("steps", 20000);
  const peakLr = numFlag("lr", 6e-4);
  const minLrFrac = 0.1;
  const warmup = numFlag("warmup", 200);
  const evalEvery = numFlag("eval-every", 500);
  const outName = strFlag("out", "gpt-big.bin");
  const resumeName = strFlag("resume", "");

  // A checkpoint's architecture is a property of its weights, not of this
  // invocation's flags: adopt it so `--resume` can never reinterpret a saved
  // model as the wrong shape.
  if (resumeName) {
    const prior = await readCheckpointMeta(join(CKPT_DIR, resumeName));
    const priorArch = prior.config.arch ?? "gpt2";
    if (priorArch !== config.arch) {
      console.log(`Resuming a "${priorArch}" checkpoint — ignoring --arch ${config.arch}`);
      config.arch = priorArch;
    }
  }

  const engine = await GpuEngine.create();
  const gpuModel = new GpuGPT(engine, config);
  const cpuModel = new GPT(config, mulberry32(1337));

  let startStep = 0;
  let bestVal = Infinity;
  if (resumeName) {
    const m = await loadCheckpoint(join(CKPT_DIR, resumeName), cpuModel);
    startStep = m.step;
    bestVal = m.bestValLoss;
    cpuToGpu(engine, cpuModel, gpuModel);
    console.log(`Resumed from ${resumeName} at step ${startStep} (best val ${bestVal.toFixed(4)})`);
  } else {
    // Seed the GPU model from a fresh CPU model so the initialization is exactly
    // the well-tested CPU init (GPT-2 scaled-residual scheme), not the GPU's.
    cpuToGpu(engine, cpuModel, gpuModel);
  }
  // Flush the queue so wgpu recycles writeBuffer staging memory. On discrete
  // GPUs (NVIDIA/Vulkan) staging lives in the ~256MB host-visible BAR heap,
  // and without a poll the belt grows until allocation fails mid-step — the
  // weight upload above alone is a large fraction of that heap. Metal's
  // unified memory never hits this, which is why it only bit on Colab.
  await engine.device.queue.onSubmittedWorkDone();

  const tokenizer = BPETokenizer.fromJSON(JSON.parse(await readFile(join(DATA_DIR, "tokenizer.json"), "utf8")));
  const trainTokens = await loadTokens("train.bin");
  const valTokens = await loadTokens("val.bin");

  const params = gpuModel.parameters();
  const mBuf = params.map((p: GpuTensor) => engine.buffer(p.size));
  const vBuf = params.map((p: GpuTensor) => engine.buffer(p.size));
  const rng = mulberry32(42 + startStep);

  const nParams = gpuModel.numParams();
  const tokensPerStep = b * t;
  console.log(`GPU pretraining a ${(nParams / 1e6).toFixed(2)}M-param GPT`);
  console.log(`  config: ${config.nLayer}L ${config.nHead}H ${config.nEmbd}E  block ${t}  vocab ${config.vocabSize}  arch ${config.arch}`);
  console.log(`  ${steps} steps, batch ${b}x${t} = ${tokensPerStep} tok/step  (${((steps * tokensPerStep) / 1e6).toFixed(1)}M tokens total)`);
  console.log(`  train ${(trainTokens.length / 1e6).toFixed(1)}M tok, val ${(valTokens.length / 1e6).toFixed(1)}M tok, peak lr ${peakLr}\n`);

  const lrAt = (step: number): number => {
    if (step < warmup) return (peakLr * (step + 1)) / warmup;
    const p = Math.min(1, (step - warmup) / Math.max(1, steps - warmup));
    return peakLr * (minLrFrac + (1 - minLrFrac) * 0.5 * (1 + Math.cos(Math.PI * p)));
  };

  const evalLoss = async (): Promise<number> => {
    let total = 0;
    const n = 16;
    for (let i = 0; i < n; i++) {
      const [inp, tgt] = sampleBatch(valTokens, b, t, mulberry32(90000 + i));
      const l = gpuModel.loss(inp, tgt, b, t);
      total += (await engine.read(l.buffer, 1))[0];
      l.freeGraph();
      for (const p of params) p.zeroGrad();
    }
    return total / n;
  };

  const start = performance.now();
  let tstep = startStep;
  for (let step = startStep; step < steps; step++) {
    for (const p of params) p.zeroGrad();
    const [inp, tgt] = sampleBatch(trainTokens, b, t, rng);
    const loss = gpuModel.loss(inp, tgt, b, t);
    loss.backward();
    tstep++;
    const lr = lrAt(step);
    const bc1 = 1 - Math.pow(0.9, tstep);
    const bc2 = 1 - Math.pow(0.999, tstep);
    for (let i = 0; i < params.length; i++) {
      engine.adamw(params[i].grad!, params[i].buffer, mBuf[i], vBuf[i], params[i].size, lr, 0.9, 0.999, 1e-8, 0.1, bc1, bc2);
    }
    // Per-step queue flush: caps the staging belt at one step's uniforms and
    // batch uploads instead of letting 20 steps of them pile up (see above).
    await engine.device.queue.onSubmittedWorkDone();

    if ((step + 1) % 20 === 0) {
      const l = (await engine.read(loss.buffer, 1))[0];
      const secs = (performance.now() - start) / 1000;
      const done = step + 1 - startStep;
      const tokPerSec = (done * tokensPerStep) / secs;
      const eta = fmtDuration(((steps - step - 1) / done) * secs);
      console.log(`step ${(step + 1).toString().padStart(6)}/${steps}  loss ${l.toFixed(4)}  lr ${lr.toExponential(2)}  ${tokPerSec.toFixed(0)} tok/s  eta ${eta}`);
    }
    loss.freeGraph();

    if ((step + 1) % evalEvery === 0 || step + 1 === steps) {
      const vl = await evalLoss();
      if (vl < bestVal) bestVal = vl;
      await gpuToCpu(engine, gpuModel, cpuModel);
      await saveCheckpoint(join(CKPT_DIR, outName), cpuModel, step + 1, bestVal);
      const prompt = tokenizer.encode("The ");
      const gen = cpuModel.generate(prompt, 40, { temperature: 0.8, topK: 40, rng });
      const text = tokenizer.decode(gen).replace(/\n/g, " ").slice(0, 140);
      console.log(`\n  eval @ ${step + 1}: val loss ${vl.toFixed(4)}  (best ${bestVal.toFixed(4)})  elapsed ${fmtDuration((performance.now() - start) / 1000)}`);
      console.log(`  sample: ${JSON.stringify(text)}\n`);
    }
  }
  console.log(`\nDone in ${fmtDuration((performance.now() - start) / 1000)}. Best val loss ${bestVal.toFixed(4)}. Saved to checkpoints/${outName}`);
}

main().catch((e) => {
  console.error(e);
  Deno.exit(1);
});
