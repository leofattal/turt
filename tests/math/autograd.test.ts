import { describe, expect, it } from "vitest";
import { Tensor } from "../../src/index.js";

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

describe("Autograd", () => {
  it("matches numeric gradients for a broadcasted element-wise expression", () => {
    const aVals = [0.5, -1.2, 0.3, 1.1, -0.4, 0.9];
    const bVals = [0.7, -0.2, 1.3];

    const lossOf = (av: number[], bv: number[]) => {
      const a = Tensor.fromArray(av, [2, 3], true);
      const b = Tensor.fromArray(bv, [3], true);
      return a.mul(b).add(a.tanh()).sum();
    };

    const a = Tensor.fromArray(aVals, [2, 3], true);
    const b = Tensor.fromArray(bVals, [3], true);
    const loss = a.mul(b).add(a.tanh()).sum();
    loss.backward();

    const numA = numericGrad((v) => lossOf(v, bVals).item(), aVals);
    const numB = numericGrad((v) => lossOf(aVals, v).item(), bVals);
    for (let i = 0; i < numA.length; i++) expect(a.grad![i]).toBeCloseTo(numA[i], 2);
    for (let i = 0; i < numB.length; i++) expect(b.grad![i]).toBeCloseTo(numB[i], 2);
  });

  it("matches numeric gradients through matmul and mean", () => {
    const aVals = [0.2, -0.5, 0.8, 1.0, -0.3, 0.6];
    const bVals = [0.4, -0.9, 1.2, 0.1, -0.7, 0.5];

    const lossOf = (av: number[], bv: number[]) => {
      const a = Tensor.fromArray(av, [2, 3], true);
      const b = Tensor.fromArray(bv, [3, 2], true);
      return a.matmul(b).relu().mean();
    };

    const a = Tensor.fromArray(aVals, [2, 3], true);
    const b = Tensor.fromArray(bVals, [3, 2], true);
    const loss = a.matmul(b).relu().mean();
    loss.backward();

    const numA = numericGrad((v) => lossOf(v, bVals).item(), aVals);
    const numB = numericGrad((v) => lossOf(aVals, v).item(), bVals);
    for (let i = 0; i < numA.length; i++) expect(a.grad![i]).toBeCloseTo(numA[i], 2);
    for (let i = 0; i < numB.length; i++) expect(b.grad![i]).toBeCloseTo(numB[i], 2);
  });

  it("handles diamond-shaped graphs (a value used twice)", () => {
    // d = c + tanh(c) with c = a*b: gradients must accumulate from both paths
    // before c's own backward runs.
    const a = Tensor.scalar(0.8, true);
    const b = Tensor.scalar(-1.5, true);
    const c = a.mul(b);
    const d = c.add(c.tanh()).sum();
    d.backward();

    const x = 0.8 * -1.5;
    const dc = 1 + (1 - Math.tanh(x) ** 2);
    expect(a.grad![0]).toBeCloseTo(dc * -1.5, 4);
    expect(b.grad![0]).toBeCloseTo(dc * 0.8, 4);
  });

  it("accumulates gradients until zeroGrad", () => {
    const a = Tensor.scalar(2, true);
    a.mulScalar(3).sum().backward();
    a.mulScalar(3).sum().backward();
    expect(a.grad![0]).toBeCloseTo(6, 5);
    a.zeroGrad();
    expect(a.grad).toBeNull();
  });
});
