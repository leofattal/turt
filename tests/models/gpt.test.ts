import { describe, expect, it } from "vitest";
import { AdamW, GPT, Tensor, archOf, clipGradNorm, mlpHidden, mulberry32 } from "../../src/index.js";
import type { GPTConfig } from "../../src/index.js";

const config = { vocabSize: 16, blockSize: 8, nLayer: 2, nHead: 2, nEmbd: 16 };

describe("GPT", () => {
  it("produces logits of shape [batch * time, vocabSize]", () => {
    const model = new GPT(config, mulberry32(1));
    const idx = Tensor.fromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    const logits = model.forward(idx);
    expect(logits.shape).toEqual([6, config.vocabSize]);
    expect(logits.toArray().every(Number.isFinite)).toBe(true);
  });

  it("starts near the uniform-prior loss of ln(vocabSize)", () => {
    const model = new GPT(config, mulberry32(2));
    const idx = Tensor.fromArray([0, 1, 2, 3, 4, 5, 6, 7], [1, 8]);
    const loss = model.loss(idx, [1, 2, 3, 4, 5, 6, 7, 0]);
    // A correctly initialized model predicts roughly uniformly at step 0.
    expect(loss.item()).toBeGreaterThan(Math.log(config.vocabSize) - 0.5);
    expect(loss.item()).toBeLessThan(Math.log(config.vocabSize) + 0.5);
  });

  it("is causal: later tokens cannot change earlier predictions", () => {
    const model = new GPT(config, mulberry32(3));
    const a = model.forward(Tensor.fromArray([1, 2, 3, 4], [1, 4])).toArray();
    // Same prefix, different final token — the first 3 rows must be identical.
    const b = model.forward(Tensor.fromArray([1, 2, 3, 9], [1, 4])).toArray();
    const prefixLen = 3 * config.vocabSize;
    for (let i = 0; i < prefixLen; i++) expect(a[i]).toBeCloseTo(b[i], 5);
    // The last row should differ, confirming the token was actually used.
    expect(a.slice(prefixLen)).not.toEqual(b.slice(prefixLen));
  });

  it("routes gradients to every parameter", () => {
    const model = new GPT(config, mulberry32(4));
    model.loss(Tensor.fromArray([1, 2, 3, 4], [1, 4]), [2, 3, 4, 5]).backward();
    for (const p of model.parameters()) {
      expect(p.grad, "every parameter should receive a gradient").not.toBeNull();
      expect(p.grad!.some((v) => v !== 0)).toBe(true);
    }
  });

  it("overfits a fixed sequence (learning works end to end)", () => {
    const rng = mulberry32(7);
    const model = new GPT(config, rng);
    const optimizer = new AdamW(model.parameters(), { lr: 0.01 });

    const sequence = [1, 2, 3, 4, 5, 6, 7, 8];
    const idx = Tensor.fromArray(sequence.slice(0, 7), [1, 7]);
    const targets = sequence.slice(1);

    const initialLoss = model.loss(idx, targets).item();
    for (let step = 0; step < 120; step++) {
      optimizer.zeroGrad();
      const loss = model.loss(idx, targets);
      loss.backward();
      clipGradNorm(model.parameters(), 1.0);
      optimizer.step();
    }
    const finalLoss = model.loss(idx, targets).item();

    expect(finalLoss).toBeLessThan(initialLoss);
    expect(finalLoss).toBeLessThan(0.05);

    // Having memorized the sequence, greedy continuation should reproduce it.
    const generated = model.generate([1], 7, { temperature: 0.01, rng });
    expect(generated).toEqual(sequence);
  });

  it("generates within the vocabulary and respects the requested length", () => {
    const model = new GPT(config, mulberry32(8));
    const out = model.generate([1, 2], 10, { temperature: 0.8, topK: 4, rng: mulberry32(9) });
    expect(out).toHaveLength(12);
    expect(out.every((id) => id >= 0 && id < config.vocabSize)).toBe(true);
  });

  it("streams tokens through onToken and stops when it returns false", () => {
    const model = new GPT(config, mulberry32(8));
    const seen: number[] = [];
    const out = model.generate([1, 2], 10, {
      temperature: 0.8,
      rng: mulberry32(9),
      onToken: (id) => {
        seen.push(id);
        if (seen.length === 3) return false;
      },
    });
    expect(seen).toHaveLength(3);
    expect(out).toEqual([1, 2, ...seen]);
  });

  it("crops context to blockSize when generating past it", () => {
    const model = new GPT(config, mulberry32(10));
    const prompt = Array.from({ length: config.blockSize }, (_, i) => i % config.vocabSize);
    const out = model.generate(prompt, 5, { rng: mulberry32(11) });
    expect(out).toHaveLength(config.blockSize + 5);
  });

  it("counts parameters", () => {
    const model = new GPT(config, mulberry32(12));
    expect(model.numParams()).toBeGreaterThan(0);
    // Weight tying means the output head adds no parameters of its own.
    const embedding = config.vocabSize * config.nEmbd + config.blockSize * config.nEmbd;
    const perBlock = 12 * config.nEmbd * config.nEmbd + 13 * config.nEmbd;
    expect(model.numParams()).toBe(embedding + config.nLayer * perBlock + 2 * config.nEmbd);
  });

  it("defaults to the gpt2 architecture, so old checkpoints keep their shape", () => {
    expect(archOf(config)).toBe("gpt2");
  });
});

describe('GPT (arch "modern")', () => {
  const modern: GPTConfig = { ...config, arch: "modern" };

  it("drops the position table and every bias", () => {
    const model = new GPT(modern, mulberry32(1));
    const { nEmbd: e, nLayer, vocabSize } = modern;
    const h = mlpHidden(modern);
    // wte only — RoPE needs no table. Per block: one gain per norm (no bias),
    // four square attention matrices (no bias), three SwiGLU matrices (no bias).
    const perBlock = 4 * e * e + 3 * e * h + 2 * e;
    expect(model.numParams()).toBe(vocabSize * e + nLayer * perBlock + e);
  });

  it("sizes SwiGLU to match the GELU block's parameter count", () => {
    // 8/3 * nEmbd across three matrices ≈ 4 * nEmbd across two.
    const wide: GPTConfig = { ...config, nEmbd: 384, arch: "modern" };
    const swiglu = 3 * 384 * mlpHidden(wide);
    const gelu = 2 * 384 * mlpHidden({ ...wide, arch: "gpt2" });
    expect(swiglu / gelu).toBeCloseTo(1, 1);
  });

  it("is causal: later tokens cannot change earlier predictions", () => {
    const model = new GPT(modern, mulberry32(3));
    const a = model.forward(Tensor.fromArray([1, 2, 3, 4], [1, 4])).toArray();
    const b = model.forward(Tensor.fromArray([1, 2, 3, 9], [1, 4])).toArray();
    const prefixLen = 3 * modern.vocabSize;
    for (let i = 0; i < prefixLen; i++) expect(a[i]).toBeCloseTo(b[i], 5);
    expect(a.slice(prefixLen)).not.toEqual(b.slice(prefixLen));
  });

  it("routes gradients to every parameter", () => {
    const model = new GPT(modern, mulberry32(4));
    model.loss(Tensor.fromArray([1, 2, 3, 4], [1, 4]), [2, 3, 4, 5]).backward();
    for (const p of model.parameters()) {
      expect(p.grad, "every parameter should receive a gradient").not.toBeNull();
      expect(p.grad!.some((v) => v !== 0)).toBe(true);
    }
  });

  it("overfits a fixed sequence (learning works end to end)", () => {
    const rng = mulberry32(7);
    const model = new GPT(modern, rng);
    const optimizer = new AdamW(model.parameters(), { lr: 0.01 });

    const sequence = [1, 2, 3, 4, 5, 6, 7, 8];
    const idx = Tensor.fromArray(sequence.slice(0, 7), [1, 7]);
    const targets = sequence.slice(1);

    for (let step = 0; step < 120; step++) {
      optimizer.zeroGrad();
      model.loss(idx, targets).backward();
      clipGradNorm(model.parameters(), 1.0);
      optimizer.step();
    }
    expect(model.loss(idx, targets).item()).toBeLessThan(0.05);
    expect(model.generate([1], 7, { temperature: 0.01, rng })).toEqual(sequence);
  });
});
