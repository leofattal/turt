import { describe, expect, it } from "vitest";
import { Tensor, mulberry32 } from "../../src/index.js";

/** Central-difference numeric gradient of a scalar function of flat values. */
function numericGrad(f: (vals: number[]) => number, vals: number[], eps = 1e-3): number[] {
  return vals.map((_, i) => {
    const plus = [...vals];
    const minus = [...vals];
    plus[i] += eps;
    minus[i] -= eps;
    return (f(plus) - f(minus)) / (2 * eps);
  });
}

/** Asserts analytic gradients of `build` match numeric ones at `vals`. */
function checkGrad(
  vals: number[],
  shape: number[],
  build: (t: Tensor) => Tensor,
  precision = 2,
): void {
  const t = Tensor.fromArray(vals, shape, true);
  build(t).backward();
  const numeric = numericGrad((v) => build(Tensor.fromArray(v, shape, true)).item(), vals);
  for (let i = 0; i < vals.length; i++) {
    expect(t.grad![i]).toBeCloseTo(numeric[i], precision);
  }
}

const rng = mulberry32(5);
const randVals = (n: number): number[] => Array.from({ length: n }, () => rng() * 2 - 1);

describe("permute", () => {
  it("reorders axes", () => {
    const t = Tensor.fromArray([1, 2, 3, 4, 5, 6], [1, 3, 2]);
    const p = t.permute([0, 2, 1]);
    expect(p.shape).toEqual([1, 2, 3]);
    expect(p.toArray()).toEqual([1, 3, 5, 2, 4, 6]);
  });

  it("is its own inverse for a transposition", () => {
    const t = Tensor.fromArray(randVals(24), [2, 3, 4]);
    const round = t.permute([2, 1, 0]).permute([2, 1, 0]);
    expect(round.toArray()).toEqual(t.toArray());
  });

  it("has correct gradients", () => {
    checkGrad(randVals(24), [2, 3, 4], (t) => t.permute([1, 0, 2]).mul(t.permute([1, 0, 2])).sum());
  });
});

describe("bmm", () => {
  it("matches per-group matmul", () => {
    // group 0: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    const a = Tensor.fromArray([1, 2, 3, 4, 1, 0, 0, 1], [2, 2, 2]);
    const b = Tensor.fromArray([5, 6, 7, 8, 9, 9, 9, 9], [2, 2, 2]);
    expect(a.bmm(b).toArray()).toEqual([19, 22, 43, 50, 9, 9, 9, 9]);
  });

  it("has correct gradients for both operands", () => {
    const aVals = randVals(12); // [2,2,3]
    const bVals = randVals(12); // [2,3,2]
    const lossOf = (av: number[], bv: number[]) =>
      Tensor.fromArray(av, [2, 2, 3], true).bmm(Tensor.fromArray(bv, [2, 3, 2], true)).tanh().sum();

    const a = Tensor.fromArray(aVals, [2, 2, 3], true);
    const b = Tensor.fromArray(bVals, [2, 3, 2], true);
    a.bmm(b).tanh().sum().backward();

    const numA = numericGrad((v) => lossOf(v, bVals).item(), aVals);
    const numB = numericGrad((v) => lossOf(aVals, v).item(), bVals);
    for (let i = 0; i < numA.length; i++) expect(a.grad![i]).toBeCloseTo(numA[i], 2);
    for (let i = 0; i < numB.length; i++) expect(b.grad![i]).toBeCloseTo(numB[i], 2);
  });
});

describe("gatherRows", () => {
  it("gathers embedding rows", () => {
    const table = Tensor.fromArray([1, 2, 3, 4, 5, 6], [3, 2]);
    expect(table.gatherRows([2, 0]).toArray()).toEqual([5, 6, 1, 2]);
  });

  it("scatter-adds gradients for repeated indices", () => {
    const table = Tensor.fromArray([1, 2, 3, 4], [2, 2], true);
    table.gatherRows([0, 0, 1]).sum().backward();
    // Row 0 was gathered twice, row 1 once.
    expect([...table.grad!]).toEqual([2, 2, 1, 1]);
  });

  it("rejects out-of-range indices", () => {
    expect(() => Tensor.zeros([2, 2]).gatherRows([5])).toThrow();
  });
});

describe("softmax", () => {
  it("produces a normalized distribution over the last dim", () => {
    const out = Tensor.fromArray([1, 2, 3, 0, 0, 0], [2, 3]).softmax();
    const rows = [out.toArray().slice(0, 3), out.toArray().slice(3, 6)];
    for (const row of rows) {
      expect(row.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 5);
      expect(row.every((v) => v > 0)).toBe(true);
    }
    expect(rows[1]).toEqual([1 / 3, 1 / 3, 1 / 3].map((v) => expect.closeTo(v, 5)));
  });

  it("is numerically stable for large logits", () => {
    const out = Tensor.fromArray([1000, 1001, 1002], [1, 3]).softmax();
    expect(out.toArray().every(Number.isFinite)).toBe(true);
    expect(out.toArray().reduce((s, v) => s + v, 0)).toBeCloseTo(1, 5);
  });

  it("has correct gradients", () => {
    checkGrad(randVals(6), [2, 3], (t) =>
      t.softmax().mul(Tensor.fromArray([1, 2, 3, 4, 5, 6], [2, 3])).sum(),
    );
  });
});

describe("maskedFillCausal", () => {
  it("zeroes attention to future positions after softmax", () => {
    const scores = Tensor.fromArray([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3]);
    const attn = scores.maskedFillCausal().softmax().toArray();
    // Row 0 attends only to itself; row 1 to the first two; row 2 to all.
    expect(attn[1]).toBeCloseTo(0, 6);
    expect(attn[2]).toBeCloseTo(0, 6);
    expect(attn[5]).toBeCloseTo(0, 6);
    expect(attn[0]).toBeCloseTo(1, 6);
    expect(attn[3] + attn[4]).toBeCloseTo(1, 5);
  });

  it("passes gradients only through unmasked positions", () => {
    const t = Tensor.fromArray(randVals(9), [1, 3, 3], true);
    t.maskedFillCausal().sum().backward();
    // Lower triangle (including diagonal) gets gradient 1; upper triangle 0.
    expect([...t.grad!]).toEqual([1, 0, 0, 1, 1, 0, 1, 1, 1]);
  });
});

describe("gelu", () => {
  it("matches known values", () => {
    const out = Tensor.fromArray([-1, 0, 1], [3]).gelu().toArray();
    expect(out[0]).toBeCloseTo(-0.1588, 3);
    expect(out[1]).toBeCloseTo(0, 6);
    expect(out[2]).toBeCloseTo(0.8412, 3);
  });

  it("has correct gradients", () => {
    checkGrad(randVals(6), [2, 3], (t) => t.gelu().sum());
  });
});

describe("crossEntropyLogits", () => {
  it("matches the analytic loss of a uniform distribution", () => {
    const loss = Tensor.zeros([2, 4]).crossEntropyLogits([0, 3]);
    expect(loss.item()).toBeCloseTo(Math.log(4), 5);
  });

  it("approaches zero for a confident correct prediction", () => {
    const loss = Tensor.fromArray([0, 20, 0, 0], [1, 4]).crossEntropyLogits([1]);
    expect(loss.item()).toBeLessThan(1e-6);
  });

  it("is stable for large logits", () => {
    const loss = Tensor.fromArray([1000, 1001, 999], [1, 3]).crossEntropyLogits([0]);
    expect(Number.isFinite(loss.item())).toBe(true);
  });

  it("has gradients equal to (softmax - onehot) / n", () => {
    const logits = Tensor.fromArray([1, 2, 3, 4], [2, 2], true);
    logits.crossEntropyLogits([0, 1]).backward();
    const numeric = numericGrad(
      (v) => Tensor.fromArray(v, [2, 2], true).crossEntropyLogits([0, 1]).item(),
      [1, 2, 3, 4],
    );
    for (let i = 0; i < 4; i++) expect(logits.grad![i]).toBeCloseTo(numeric[i], 3);
  });
});
