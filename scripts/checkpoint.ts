/**
 * Checkpoint format: a JSON header (config, step, optimizer state shapes)
 * followed by raw Float32 parameter blocks. Parameters are written in
 * `model.parameters()` order, which is deterministic for a given config;
 * shapes are recorded and verified on load so a mismatch fails loudly rather
 * than silently loading garbage.
 */

import { readFile, writeFile } from "node:fs/promises";
import type { GPT, GPTConfig } from "../src/models/gpt.js";

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

export async function loadCheckpoint(path: string, model: GPT): Promise<CheckpointMeta> {
  const buffer = await readFile(path);
  if (buffer.subarray(0, 8).toString("ascii").replace(/\0+$/, "") !== MAGIC) {
    throw new Error(`${path} is not a Turt checkpoint`);
  }
  const headerLength = buffer.readUInt32LE(8);
  const meta = JSON.parse(buffer.subarray(12, 12 + headerLength).toString("utf8")) as CheckpointMeta;

  const params = model.parameters();
  if (params.length !== meta.shapes.length) {
    throw new Error(`Checkpoint has ${meta.shapes.length} params, model has ${params.length}`);
  }

  let offset = 12 + headerLength;
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
