/**
 * Samples text from a trained checkpoint.
 *
 * Run: pnpm generate [--prompt "It was a"] [--tokens 200]
 *                    [--temperature 0.8] [--top-k 40] [--ckpt checkpoints/gpt.bin]
 */

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { GPT } from "../src/models/gpt.js";
import { mulberry32 } from "../src/math/random.js";
import { BPETokenizer } from "../src/tokenizer/bpe.js";
import { loadCheckpoint, type CheckpointMeta } from "./checkpoint.js";

const ROOT = join(import.meta.dirname, "..");

const flag = (name: string, fallback: string): string => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
};

async function main(): Promise<void> {
  const ckptPath = join(ROOT, flag("ckpt", "checkpoints/gpt.bin"));
  const prompt = flag("prompt", "It was a");
  const maxTokens = Number(flag("tokens", "200"));
  const temperature = Number(flag("temperature", "0.8"));
  const topK = Number(flag("top-k", "40"));
  const seed = Number(flag("seed", String(Date.now() % 100000)));

  const tokenizer = BPETokenizer.fromJSON(
    JSON.parse(await readFile(join(ROOT, "data", "tokenizer.json"), "utf8")),
  );

  // The header carries the config, so the model is reconstructed exactly.
  const buffer = await readFile(ckptPath);
  const headerLength = buffer.readUInt32LE(8);
  const meta = JSON.parse(buffer.subarray(12, 12 + headerLength).toString("utf8")) as CheckpointMeta;

  const model = new GPT(meta.config, mulberry32(seed));
  await loadCheckpoint(ckptPath, model);

  console.log(
    `${ckptPath} — step ${meta.step}, val loss ${meta.bestValLoss.toFixed(4)}, ` +
      `${(model.numParams() / 1e6).toFixed(2)}M params\n`,
  );

  const ids = model.generate(tokenizer.encode(prompt), maxTokens, {
    temperature,
    topK,
    rng: mulberry32(seed),
  });
  console.log(tokenizer.decode(ids));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
