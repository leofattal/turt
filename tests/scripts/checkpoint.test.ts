import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { GPT, Tensor, mulberry32 } from "../../src/index.js";
import { loadCheckpoint, saveCheckpoint } from "../../scripts/checkpoint.js";
import { flattenGrads, flattenParams, loadGrads, loadParams } from "../../scripts/params.js";

const config = { vocabSize: 32, blockSize: 8, nLayer: 2, nHead: 2, nEmbd: 16 };
let dir: string;

beforeAll(async () => {
  dir = await mkdtemp(join(tmpdir(), "turt-ckpt-"));
});
afterAll(async () => {
  await rm(dir, { recursive: true, force: true });
});

describe("checkpointing", () => {
  it("round-trips weights exactly and reproduces logits", async () => {
    const original = new GPT(config, mulberry32(1));
    const idx = Tensor.fromArray([1, 2, 3, 4], [1, 4]);
    const before = original.forward(idx).toArray();

    const path = join(dir, "gpt.bin");
    await saveCheckpoint(path, original, 42, 1.234);

    // A differently-seeded model starts with different weights...
    const restored = new GPT(config, mulberry32(999));
    expect(restored.forward(idx).toArray()).not.toEqual(before);

    // ...and after loading, it must be bit-for-bit the original.
    const meta = await loadCheckpoint(path, restored);
    expect(meta.step).toBe(42);
    expect(meta.bestValLoss).toBeCloseTo(1.234, 6);
    expect(meta.config).toEqual(config);
    expect(restored.forward(idx).toArray()).toEqual(before);
  });

  it("rejects a checkpoint whose shapes do not match the model", async () => {
    const path = join(dir, "mismatch.bin");
    await saveCheckpoint(path, new GPT(config, mulberry32(2)), 1, 0);

    const wider = new GPT({ ...config, nEmbd: 32 }, mulberry32(2));
    await expect(loadCheckpoint(path, wider)).rejects.toThrow(/mismatch|params/i);
  });

  it("rejects a file that is not a Turt checkpoint", async () => {
    const { writeFile } = await import("node:fs/promises");
    const path = join(dir, "garbage.bin");
    await writeFile(path, Buffer.alloc(256, 7));
    await expect(loadCheckpoint(path, new GPT(config, mulberry32(3)))).rejects.toThrow(/not a Turt/i);
  });
});

describe("flat parameter views (data-parallel transport)", () => {
  it("round-trips parameters through a flat array", () => {
    const source = new GPT(config, mulberry32(4));
    const target = new GPT(config, mulberry32(5));
    const flat = flattenParams(source.parameters());
    expect(flat.length).toBe(source.numParams());

    loadParams(target.parameters(), flat);
    const idx = Tensor.fromArray([1, 2, 3], [1, 3]);
    expect(target.forward(idx).toArray()).toEqual(source.forward(idx).toArray());
  });

  it("round-trips gradients through a flat array", () => {
    const model = new GPT(config, mulberry32(6));
    model.loss(Tensor.fromArray([1, 2, 3], [1, 3]), [2, 3, 4]).backward();

    const grads = flattenGrads(model.parameters());
    expect(grads.some((v) => v !== 0)).toBe(true);

    const replica = new GPT(config, mulberry32(7));
    loadGrads(replica.parameters(), grads);
    const params = model.parameters();
    replica.parameters().forEach((p, i) => {
      expect([...p.grad!]).toEqual([...params[i].grad!]);
    });
  });

  it("averages sharded gradients back to the full-batch gradient", () => {
    // The core claim of data-parallel training: a weighted mean of per-shard
    // gradients equals the gradient of the full batch.
    const inputs = [1, 2, 3, 4, 5, 6, 7, 8];
    const targets = [2, 3, 4, 5, 6, 7, 8, 9];

    const full = new GPT(config, mulberry32(8));
    full.loss(Tensor.fromArray(inputs, [2, 4]), targets).backward();
    const expected = flattenGrads(full.parameters());

    // Same weights, but one sequence at a time (two shards of one row each).
    const sharded = new GPT(config, mulberry32(8));
    const accumulated = new Float32Array(expected.length);
    for (let row = 0; row < 2; row++) {
      for (const p of sharded.parameters()) p.zeroGrad();
      const slice = inputs.slice(row * 4, row * 4 + 4);
      sharded.loss(Tensor.fromArray(slice, [1, 4]), targets.slice(row * 4, row * 4 + 4)).backward();
      const g = flattenGrads(sharded.parameters());
      for (let i = 0; i < g.length; i++) accumulated[i] += g[i] * 0.5; // weight = rows/batch
    }

    for (let i = 0; i < expected.length; i++) {
      expect(accumulated[i]).toBeCloseTo(expected[i], 5);
    }
  });
});
