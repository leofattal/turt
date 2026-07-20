/**
 * Checkpoint format: a JSON header (config, step, optimizer state shapes)
 * followed by raw Float32 parameter blocks. Parameters are written in
 * `model.parameters()` order, which is deterministic for a given config;
 * shapes are recorded and verified on load so a mismatch fails loudly rather
 * than silently loading garbage.
 */

import { readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { GPT, GPTConfig } from "../src/models/gpt.js";
import { BPETokenizer } from "../src/tokenizer/bpe.js";

export interface CheckpointMeta {
  config: GPTConfig;
  step: number;
  bestValLoss: number;
  shapes: number[][];
}

const MAGIC = "TURTCKPT";

export async function saveCheckpoint(
  path: string,
  model: GPT,
  step: number,
  bestValLoss: number,
): Promise<void> {
  const params = model.parameters();
  const meta: CheckpointMeta = {
    config: model.config,
    step,
    bestValLoss,
    shapes: params.map((p) => p.shape),
  };
  const header = Buffer.from(JSON.stringify(meta), "utf8");
  const lengthPrefix = Buffer.alloc(8);
  lengthPrefix.write(MAGIC, 0, "ascii");

  const headerLength = Buffer.alloc(4);
  headerLength.writeUInt32LE(header.length, 0);

  const blocks = params.map((p) => Buffer.from(p.data.buffer, p.data.byteOffset, p.data.byteLength));
  await writeFile(path, Buffer.concat([lengthPrefix, headerLength, header, ...blocks]));
}

function parseHeader(buffer: Buffer, path: string): { meta: CheckpointMeta; dataOffset: number } {
  if (buffer.subarray(0, 8).toString("ascii").replace(/\0+$/, "") !== MAGIC) {
    throw new Error(`${path} is not a Turt checkpoint`);
  }
  const headerLength = buffer.readUInt32LE(8);
  const meta = JSON.parse(buffer.subarray(12, 12 + headerLength).toString("utf8")) as CheckpointMeta;
  return { meta, dataOffset: 12 + headerLength };
}

/**
 * Reads only the header. Callers that must construct a model *before* loading it
 * (the trainer needs the config to build one) use this to recover the config —
 * notably `config.arch`, which decides the parameter layout.
 */
export async function readCheckpointMeta(path: string): Promise<CheckpointMeta> {
  return parseHeader(await readFile(path), path).meta;
}

/**
 * Loads the BPE tokenizer whose vocabulary matches a checkpoint's `vocabSize`.
 *
 * Rebuilding the data pipeline with a different vocab replaces
 * `data/tokenizer.json` and silently orphans every checkpoint trained on the
 * old one — ids then index the wrong embeddings (gibberish, or a crash when an
 * id exceeds the table). Older tokenizers are therefore kept alongside it as
 * `data/tokenizer-<vocab>.json`, and this picks whichever file matches.
 */
export async function loadMatchingTokenizer(
  dataDir: string,
  vocabSize: number,
  explicitPath?: string,
): Promise<BPETokenizer> {
  const candidates = explicitPath
    ? [explicitPath]
    : [join(dataDir, "tokenizer.json"), join(dataDir, `tokenizer-${vocabSize}.json`)];
  for (const path of candidates) {
    let tokenizer: BPETokenizer;
    try {
      tokenizer = BPETokenizer.fromJSON(JSON.parse(await readFile(path, "utf8")));
    } catch {
      continue; // missing or unreadable candidate; try the next
    }
    if (tokenizer.vocabSize === vocabSize) return tokenizer;
  }
  throw new Error(
    `No tokenizer with vocab ${vocabSize} found (tried ${candidates.join(", ")}). ` +
      `The checkpoint was trained with a tokenizer that no longer exists — ` +
      `regenerate it with the matching prepare-data settings and save it as ` +
      `data/tokenizer-${vocabSize}.json.`,
  );
}

export async function loadCheckpoint(path: string, model: GPT): Promise<CheckpointMeta> {
  const buffer = await readFile(path);
  const { meta, dataOffset } = parseHeader(buffer, path);

  const params = model.parameters();
  if (params.length !== meta.shapes.length) {
    throw new Error(`Checkpoint has ${meta.shapes.length} params, model has ${params.length}`);
  }

  let offset = dataOffset;
  for (let i = 0; i < params.length; i++) {
    const p = params[i];
    const expected = meta.shapes[i];
    if (p.shape.length !== expected.length || p.shape.some((d, j) => d !== expected[j])) {
      throw new Error(
        `Param ${i} shape mismatch: checkpoint [${expected.join(",")}], model [${p.shape.join(",")}]`,
      );
    }
    for (let j = 0; j < p.size; j++) {
      p.data[j] = buffer.readFloatLE(offset + j * 4);
    }
    offset += p.size * 4;
  }
  return meta;
}
