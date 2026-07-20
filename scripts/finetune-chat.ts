/**
 * Instruction-tunes the pretrained GPT into the "Turt" chatbot, on the GPU.
 *
 *   deno run --unstable-webgpu --unstable-sloppy-imports --allow-read --allow-write \
 *     scripts/finetune-chat.ts [--steps 3000] [--lr 2e-4] [--base gpt.bin] [--out chat.bin]
 *
 * Loads the pretrained novel checkpoint into the GPU-resident GpuGPT, fine-tunes
 * it on the chat corpus with a low learning rate, and periodically saves a
 * CPU-format checkpoint (checkpoints/chat.bin) by reading the GPU weights back
 * into a CPU GPT. Fine-tuning on the GPU is ~50x faster than the CPU trainer, so
 * this finishes in minutes.
 */

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { GPT, type GPTConfig } from "../src/models/gpt.ts";
import { GpuGPT } from "../src/models/gpu-gpt.ts";
import { GpuEngine, type GpuTensor } from "../src/infer/webgpu/autograd.ts";
import { mulberry32 } from "../src/math/random.ts";
import { BPETokenizer } from "../src/tokenizer/bpe.ts";
import { loadCheckpoint, readCheckpointMeta, saveCheckpoint } from "./checkpoint.ts";

const ROOT = join(import.meta.dirname, "..");
const DATA_DIR = join(ROOT, "data");
const CKPT_DIR = join(ROOT, "checkpoints");

const flag = (name: string, fallback: number): number => {
  const i = Deno.args.indexOf(`--${name}`);
  return i >= 0 && Deno.args[i + 1] ? Number(Deno.args[i + 1]) : fallback;
};

const strFlag = (name: string, fallback: string): string => {
  const i = Deno.args.indexOf(`--${name}`);
  return i >= 0 && Deno.args[i + 1] ? Deno.args[i + 1] : fallback;
};

async function loadTokens(name: string): Promise<Uint16Array> {
  const buf = await readFile(join(DATA_DIR, name));
  return new Uint16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2);
}

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

/** Copies CPU params into GPU params (same order). */
function cpuToGpu(engine: GpuEngine, cpu: GPT, gpu: GpuGPT): void {
  const cp = cpu.parameters();
  const gp = gpu.parameters();
  for (let i = 0; i < cp.length; i++) engine.device.queue.writeBuffer(gp[i].buffer, 0, cp[i].data);
}

/** Reads GPU params back into the CPU model, for checkpointing. */
async function gpuToCpu(engine: GpuEngine, gpu: GpuGPT, cpu: GPT): Promise<void> {
  const cp = cpu.parameters();
  const gp = gpu.parameters();
  for (let i = 0; i < cp.length; i++) cp[i].data.set(await engine.read(gp[i].buffer, gp[i].size));
}

async function main(): Promise<void> {
  const steps = flag("steps", 3000);
  const peakLr = flag("lr", 2e-4);
  const warmup = 100;
  const b = 16;
  const baseName = strFlag("base", "gpt.bin");
  const outName = strFlag("out", "chat.bin");

  // Read the pretrained checkpoint's config, then build matching models. The
  // config carries `arch`, so the fine-tune inherits the base model's shape.
  const meta = await readCheckpointMeta(join(CKPT_DIR, baseName));
  const config: GPTConfig = meta.config;
  const t = config.blockSize;

  const engine = await GpuEngine.create();
  const cpuModel = new GPT(config, mulberry32(0));
  await loadCheckpoint(join(CKPT_DIR, baseName), cpuModel);
  const gpuModel = new GpuGPT(engine, config);
  cpuToGpu(engine, cpuModel, gpuModel);

  const tokenizer = BPETokenizer.fromJSON(JSON.parse(await readFile(join(DATA_DIR, "tokenizer.json"), "utf8")));
  const trainTokens = await loadTokens("chat-train.bin");
  const valTokens = await loadTokens("chat-val.bin");

  const params = gpuModel.parameters();
  const mBuf = params.map((p: GpuTensor) => engine.buffer(p.size));
  const vBuf = params.map((p: GpuTensor) => engine.buffer(p.size));
  const rng = mulberry32(7);

  console.log(`Fine-tuning ${(gpuModel.numParams() / 1e6).toFixed(2)}M-param GPT into a chatbot`);
  console.log(`  ${steps} steps, batch ${b}x${t}, lr ${peakLr}, chat data ${(trainTokens.length / 1e6).toFixed(2)}M tokens\n`);

  const lrAt = (step: number): number => {
    if (step < warmup) return (peakLr * (step + 1)) / warmup;
    const p = (step - warmup) / Math.max(1, steps - warmup);
    return peakLr * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * p)));
  };

  const evalLoss = async (): Promise<number> => {
    let total = 0;
    const n = 8;
    for (let i = 0; i < n; i++) {
      const [inp, tgt] = sampleBatch(valTokens, b, t, mulberry32(1000 + i));
      total += (await engine.read(gpuModel.loss(inp, tgt, b, t).buffer, 1))[0];
      for (const p of params) p.zeroGrad();
    }
    return total / n;
  };

  const start = performance.now();
  let tstep = 0;
  for (let step = 0; step < steps; step++) {
    for (const p of params) p.zeroGrad();
    const [inp, tgt] = sampleBatch(trainTokens, b, t, rng);
    const loss = gpuModel.loss(inp, tgt, b, t);
    loss.backward();
    tstep++;
    const lr = lrAt(step);
    const bc1 = 1 - Math.pow(0.9, tstep);
    const bc2 = 1 - Math.pow(0.999, tstep);
    for (let i = 0; i < params.length; i++) {
      engine.adamw(params[i].grad!, params[i].buffer, mBuf[i], vBuf[i], params[i].size, lr, 0.9, 0.999, 1e-8, 0.05, bc1, bc2);
    }

    if ((step + 1) % 100 === 0) {
      const l = (await engine.read(loss.buffer, 1))[0];
      const secs = (performance.now() - start) / 1000;
      const tokPerSec = ((step + 1) * b * t) / secs;
      console.log(`step ${step + 1}/${steps}  loss ${l.toFixed(4)}  lr ${lr.toExponential(2)}  ${tokPerSec.toFixed(0)} tok/s`);
    }

    if ((step + 1) % 500 === 0 || step + 1 === steps) {
      const vl = await evalLoss();
      await gpuToCpu(engine, gpuModel, cpuModel);
      await saveCheckpoint(join(CKPT_DIR, outName), cpuModel, step + 1, vl);
      // Peek at a sample continuation (raw; the chat CLI adds stop handling).
      const prompt = tokenizer.encode("User: Hello!\nTurt:");
      const gen = cpuModel.generate(prompt, 24, { temperature: 0.7, topK: 40, rng });
      const text = tokenizer.decode(gen).split("\n").slice(0, 2).join(" / ");
      console.log(`  eval val loss ${vl.toFixed(4)} — saved ${outName}`);
      console.log(`  sample: ${JSON.stringify(text)}\n`);
    }
  }
  console.log(`Done in ${((performance.now() - start) / 1000 / 60).toFixed(1)} min. Chatbot saved to checkpoints/${outName}`);
}

main().catch((e) => {
  console.error(e);
  Deno.exit(1);
});
