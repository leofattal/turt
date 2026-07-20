import { describe, expect, it } from "vitest";
import {
  Adam,
  LayerNorm,
  Linear,
  RMSNorm,
  ReLU,
  SGD,
  Sequential,
  Tensor,
  Trainer,
  clipGradNorm,
  mseLoss,
  mulberry32,
} from "../../src/index.js";

describe("Linear regression end-to-end", () => {
  it("learns y = 2x + 1", () => {
    const rng = mulberry32(7);
    const n = 64;
    const xs = new Array(n).fill(0).map(() => rng() * 2 - 1);
    const input = Tensor.fromArray(xs, [n, 1]);
    const target = Tensor.fromArray(
      xs.map((x) => 2 * x + 1),
      [n, 1],
    );

    const model = new Linear(1, 1, { rng });
    const trainer = new Trainer(model, new Adam(model.parameters(), { lr: 0.05 }));
    const finalLoss = trainer.fit([{ input, target }], { epochs: 300 });

    expect(finalLoss).toBeLessThan(1e-3);
    expect(model.weight.data[0]).toBeCloseTo(2, 1);
    expect(model.bias!.data[0]).toBeCloseTo(1, 1);
  });

  it("fits a small MLP with SGD + momentum on a nonlinear target", () => {
    const rng = mulberry32(11);
    const n = 128;
    const xs = new Array(n).fill(0).map(() => rng() * 2 - 1);
    const input = Tensor.fromArray(xs, [n, 1]);
    const target = Tensor.fromArray(
      xs.map((x) => Math.abs(x)),
      [n, 1],
    );

    const model = new Sequential(new Linear(1, 16, { rng }), new ReLU(), new Linear(16, 1, { rng }));
    const optimizer = new SGD(model.parameters(), { lr: 0.05, momentum: 0.9 });
    const trainer = new Trainer(model, optimizer);
    const finalLoss = trainer.fit([{ input, target }], { epochs: 400 });

    expect(finalLoss).toBeLessThan(0.01);
  });
});

describe("LayerNorm", () => {
  it("normalizes the last dimension to zero mean and unit variance", () => {
    const rng = mulberry32(3);
    const x = Tensor.randn([4, 8], { rng });
    const out = new LayerNorm(8).forward(x);
    for (let row = 0; row < 4; row++) {
      const vals = out.toArray().slice(row * 8, (row + 1) * 8);
      const mean = vals.reduce((s, v) => s + v, 0) / 8;
      const variance = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / 8;
      expect(mean).toBeCloseTo(0, 4);
      expect(variance).toBeCloseTo(1, 2);
    }
  });
});

describe("Gradient clipping", () => {
  it("scales the global gradient norm down to maxNorm", () => {
    const p = Tensor.fromArray([3, 4], [2], true);
    mseLoss(p.mulScalar(10), Tensor.zeros([2])).backward();
    const before = clipGradNorm([p], 1);
    expect(before).toBeGreaterThan(1);
    const norm = Math.hypot(p.grad![0], p.grad![1]);
    expect(norm).toBeCloseTo(1, 4);
  });
});

describe("RMSNorm", () => {
  it("scales each row to unit root-mean-square", () => {
    const rng = mulberry32(3);
    const out = new RMSNorm(8).forward(Tensor.randn([4, 8], { rng }));
    for (let row = 0; row < 4; row++) {
      const vals = out.toArray().slice(row * 8, (row + 1) * 8);
      const rms = Math.sqrt(vals.reduce((s, v) => s + v * v, 0) / 8);
      expect(rms).toBeCloseTo(1, 3);
    }
  });

  it("does not center the row, unlike LayerNorm", () => {
    // Every value positive: LayerNorm would force mean 0, RMSNorm must not.
    const x = Tensor.fromArray([1, 2, 3, 4], [1, 4]);
    const rmsMean = new RMSNorm(4).forward(x).toArray().reduce((s, v) => s + v, 0) / 4;
    const lnMean = new LayerNorm(4).forward(x).toArray().reduce((s, v) => s + v, 0) / 4;
    expect(lnMean).toBeCloseTo(0, 5);
    expect(rmsMean).toBeGreaterThan(0.5);
  });

  it("has a gain but no bias", () => {
    const params = new RMSNorm(8).parameters();
    expect(params).toHaveLength(1);
    expect(params[0].toArray()).toEqual(new Array(8).fill(1));
  });
});
