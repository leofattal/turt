/**
 * Trains the GPT on the tokenized Gutenberg corpus.
 *
 * Batches are sampled at random offsets from the token array, so the corpus can
 * be far larger than a single run traverses — the model simply never sees the
 * same window twice.
 *
 * Training is data-parallel across worker threads: the batch is split into
 * micro-batches, each worker runs forward/backward on its slice against a
 * replica of the model, and the main thread averages the gradients and takes
 * the optimizer step. That is mathematically identical to a single-threaded run
 * over the full batch, but uses every core.
 *
 * Run: pnpm train [--steps 20000] [--batch 32] [--block 128] [--layer 4]
 *                 [--head 4] [--embd 128] [--lr 3e-4] [--workers 8] [--resume]
 */

import { readFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
import { Worker } from "node:worker_threads";
import { cpus } from "node:os";
import { GPT, type GPTConfig } from "../src/models/gpt.js";
import { Tensor } from "../src/math/tensor.js";
import { mulberry32, type Rng } from "../src/math/random.js";
import { AdamW } from "../src/optim/adam.js";
import { clipGradNorm } from "../src/optim/clip.js";
import { BPETokenizer } from "../src/tokenizer/bpe.js";
import { saveCheckpoint, loadCheckpoint } from "./checkpoint.js";
import { flattenParams, loadGrads, totalSize } from "./params.js";
import type { StepRequest, StepResponse, WorkerInit } from "./train-worker.js";

const DATA_DIR = join(import.meta.dirname, "..", "data");
const CKPT_DIR = join(import.meta.dirname, "..", "checkpoints");

const flag = (name: string, fallback: number): number => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? Number(process.argv[i + 1]) : fallback;
};
const has = (name: string): boolean => process.argv.includes(`--${name}`);

async function loadTokens(name: string): Promise<Uint16Array> {
  const buffer = await readFile(join(DATA_DIR, name));
  return new Uint16Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 2);
}

interface Batch {
  inputs: Float32Array;
  targets: Int32Array;
}

/** Samples `batchSize` random (context, next-token) windows from the corpus. */
function sampleBatch(tokens: Uint16Array, batchSize: number, blockSize: number, rng: Rng): Batch {
  const inputs = new Float32Array(batchSize * blockSize);
  const targets = new Int32Array(batchSize * blockSize);
  for (let b = 0; b < batchSize; b++) {
    const start = Math.floor(rng() * (tokens.length - blockSize - 1));
    for (let t = 0; t < blockSize; t++) {
      inputs[b * blockSize + t] = tokens[start + t];
      targets[b * blockSize + t] = tokens[start + t + 1];
    }
  }
  return { inputs, targets };
}

/** A worker replica plus a promise-based request/response channel. */
class Replica {
  private readonly worker: Worker;
  private pending: ((response: StepResponse) => void) | null = null;

  constructor(config: GPTConfig, seed: number) {
    const init: WorkerInit = { config, seed };
    this.worker = new Worker(new URL("./train-worker.ts", import.meta.url), { workerData: init });
    this.worker.on("message", (response: StepResponse) => {
      const resolve = this.pending;
      this.pending = null;
      resolve?.(response);
    });
    this.worker.on("error", (error) => {
      console.error("worker error:", error);
      process.exit(1);
    });
  }

  step(params: Float32Array, batch: Batch, batchSize: number, blockSize: number): Promise<StepResponse> {
    return new Promise((resolve) => {
      this.pending = resolve;
      // `params` is copied by structured clone; the worker must not mutate ours.
      const request: StepRequest = {
        params,
        inputs: batch.inputs,
        targets: batch.targets,
        batchSize,
        blockSize,
      };
      this.worker.postMessage(request);
    });
  }

  async close(): Promise<void> {
    await this.worker.terminate();
  }
}

/** Splits a batch into `n` contiguous micro-batches. */
function shard(batch: Batch, batchSize: number, blockSize: number, n: number): Array<Batch & { size: number }> {
  const shards: Array<Batch & { size: number }> = [];
  const base = Math.floor(batchSize / n);
  const remainder = batchSize % n;
  let row = 0;
  for (let i = 0; i < n; i++) {
    const size = base + (i < remainder ? 1 : 0);
    if (size === 0) continue;
    const from = row * blockSize;
    const to = (row + size) * blockSize;
    shards.push({
      size,
      inputs: batch.inputs.slice(from, to),
      targets: batch.targets.slice(from, to),
    });
    row += size;
  }
  return shards;
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds)) return "?";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return h > 0 ? `${h}h${String(m).padStart(2, "0")}m` : `${m}m`;
}

/** Linear warmup, then cosine decay to 10% of the peak learning rate. */
function lrAt(step: number, totalSteps: number, peakLr: number, warmup: number): number {
  if (step < warmup) return (peakLr * (step + 1)) / warmup;
  const progress = Math.min(1, (step - warmup) / Math.max(1, totalSteps - warmup));
  return peakLr * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * progress)));
}

async function main(): Promise<void> {
  const meta = JSON.parse(await readFile(join(DATA_DIR, "meta.json"), "utf8"));
  const tokenizer = BPETokenizer.fromJSON(
    JSON.parse(await readFile(join(DATA_DIR, "tokenizer.json"), "utf8")),
  );
  const trainTokens = await loadTokens("train.bin");
  const valTokens = await loadTokens("val.bin");
  await mkdir(CKPT_DIR, { recursive: true });

  const config: GPTConfig = {
    vocabSize: meta.vocabSize,
    blockSize: flag("block", 128),
    nLayer: flag("layer", 4),
    nHead: flag("head", 4),
    nEmbd: flag("embd", 128),
  };
  const batchSize = flag("batch", 32);
  const totalSteps = flag("steps", 20000);
  const peakLr = flag("lr", 3e-4);
  const warmup = flag("warmup", 200);
  const evalEvery = flag("eval-every", 250);
  const evalBatches = flag("eval-batches", 8);
  // Leave two cores for the main thread and the OS.
  const workerCount = Math.max(1, Math.min(flag("workers", cpus().length - 2), batchSize));

  const rng = mulberry32(1337);
  const model = new GPT(config, rng);
  const params = model.parameters();
  const optimizer = new AdamW(params, { lr: peakLr, weightDecay: 0.1 });

  let startStep = 0;
  let bestValLoss = Infinity;
  const ckptPath = join(CKPT_DIR, "gpt.bin");
  if (has("resume")) {
    const loaded = await loadCheckpoint(join(CKPT_DIR, "gpt-last.bin"), model);
    startStep = loaded.step;
    bestValLoss = loaded.bestValLoss;
    console.log(`Resumed from step ${startStep} (best val loss ${bestValLoss.toFixed(4)})\n`);
  }

  const replicas = Array.from({ length: workerCount }, (_, i) => new Replica(config, 1000 + i));
  const tokensPerStep = batchSize * config.blockSize;
  const gradScratch = new Float32Array(totalSize(params));

  console.log(`Turt GPT
  params       ${(model.numParams() / 1e6).toFixed(2)}M
  layers       ${config.nLayer}   heads ${config.nHead}   embd ${config.nEmbd}   block ${config.blockSize}
  vocab        ${config.vocabSize}
  batch        ${batchSize} x ${config.blockSize} = ${tokensPerStep} tokens/step
  workers      ${workerCount} (data-parallel)
  data         ${(trainTokens.length / 1e6).toFixed(2)}M train / ${(valTokens.length / 1e6).toFixed(2)}M val tokens
  steps        ${totalSteps}  (${((totalSteps * tokensPerStep) / trainTokens.length).toFixed(2)} epochs)
  lr           ${peakLr} peak, ${warmup} warmup, cosine decay
`);

  const runStart = Date.now();
  let windowStart = Date.now();
  let windowLoss = 0;
  let windowSteps = 0;

  /** Runs one forward/backward over `batch`, spread across replicas. */
  async function distributedBackward(batch: Batch): Promise<number> {
    const flat = flattenParams(params);
    const shards = shard(batch, batchSize, config.blockSize, replicas.length);
    const responses = await Promise.all(
      shards.map((s, i) => replicas[i].step(flat, s, s.size, config.blockSize)),
    );

    // Each worker's loss and gradient is a mean over its own rows, so weight by
    // row count to recover the exact full-batch mean.
    gradScratch.fill(0);
    let loss = 0;
    for (let i = 0; i < responses.length; i++) {
      const weight = shards[i].size / batchSize;
      const grads = responses[i].grads;
      for (let j = 0; j < gradScratch.length; j++) gradScratch[j] += grads[j] * weight;
      loss += responses[i].loss * weight;
    }
    loadGrads(params, gradScratch);
    return loss;
  }

  /** Mean loss over held-out batches, evaluated on the main thread. */
  function estimateValLoss(): number {
    const evalRng = mulberry32(99); // fixed windows keep val loss comparable across steps
    let total = 0;
    for (let i = 0; i < evalBatches; i++) {
      const { inputs, targets } = sampleBatch(valTokens, batchSize, config.blockSize, evalRng);
      const idx = new Tensor(inputs, [batchSize, config.blockSize]);
      total += model.loss(idx, targets).item();
    }
    for (const p of params) p.zeroGrad(); // eval built a graph; drop its gradients
    return total / evalBatches;
  }

  for (let step = startStep; step < totalSteps; step++) {
    optimizer.lr = lrAt(step, totalSteps, peakLr, warmup);
    optimizer.zeroGrad();

    const batch = sampleBatch(trainTokens, batchSize, config.blockSize, rng);
    const loss = await distributedBackward(batch);

    clipGradNorm(params, 1.0);
    optimizer.step();

    windowLoss += loss;
    windowSteps++;

    if ((step + 1) % 10 === 0) {
      const elapsed = (Date.now() - windowStart) / 1000;
      const stepsPerSec = windowSteps / elapsed;
      console.log(
        `step ${String(step + 1).padStart(6)}/${totalSteps}  ` +
          `loss ${(windowLoss / windowSteps).toFixed(4)}  ` +
          `lr ${optimizer.lr.toExponential(2)}  ` +
          `${(stepsPerSec * tokensPerStep).toFixed(0)} tok/s  ` +
          `eta ${formatDuration((totalSteps - step - 1) / stepsPerSec)}`,
      );
      windowLoss = 0;
      windowSteps = 0;
      windowStart = Date.now();
    }

    if ((step + 1) % evalEvery === 0 || step + 1 === totalSteps) {
      const valLoss = estimateValLoss();
      console.log(
        `\n  eval @ ${step + 1}: val loss ${valLoss.toFixed(4)}  ` +
          `(best ${Math.min(bestValLoss, valLoss).toFixed(4)})  ` +
          `elapsed ${formatDuration((Date.now() - runStart) / 1000)}`,
      );

      const sample = model.generate(tokenizer.encode("The "), 60, {
        temperature: 0.8,
        topK: 40,
        rng,
      });
      for (const p of params) p.zeroGrad();
      console.log(`  sample: ${JSON.stringify(tokenizer.decode(sample))}\n`);

      if (valLoss < bestValLoss) {
        bestValLoss = valLoss;
        await saveCheckpoint(ckptPath, model, step + 1, bestValLoss);
      }
      // Always keep a resumable "last" checkpoint, even if val loss regressed.
      await saveCheckpoint(join(CKPT_DIR, "gpt-last.bin"), model, step + 1, bestValLoss);
      windowStart = Date.now(); // don't charge eval time to the next throughput window
    }
  }

  await Promise.all(replicas.map((r) => r.close()));
  console.log(
    `\nDone in ${formatDuration((Date.now() - runStart) / 1000)}. Best val loss ${bestValLoss.toFixed(4)}.`,
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
