import { describe, expect, it } from "vitest";
import { Tensor, broadcastShapes } from "../../src/index.js";

describe("Tensor basics", () => {
  it("creates tensors with matching shape and data", () => {
    const t = Tensor.fromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    expect(t.shape).toEqual([2, 3]);
    expect(t.size).toBe(6);
    expect(t.toArray()).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("rejects mismatched data/shape", () => {
    expect(() => new Tensor([1, 2, 3], [2, 2])).toThrow();
  });

  it("broadcasts shapes per NumPy rules", () => {
    expect(broadcastShapes([2, 3], [3])).toEqual([2, 3]);
    expect(broadcastShapes([4, 1], [1, 5])).toEqual([4, 5]);
    expect(() => broadcastShapes([2, 3], [4])).toThrow();
  });

  it("adds with broadcasting", () => {
    const a = Tensor.fromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    const b = Tensor.fromArray([10, 20, 30], [3]);
    expect(a.add(b).toArray()).toEqual([11, 22, 33, 14, 25, 36]);
  });

  it("computes matmul", () => {
    const a = Tensor.fromArray([1, 2, 3, 4], [2, 2]);
    const b = Tensor.fromArray([5, 6, 7, 8], [2, 2]);
    expect(a.matmul(b).toArray()).toEqual([19, 22, 43, 50]);
  });

  it("sums along an axis", () => {
    const a = Tensor.fromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    expect(a.sum(0).toArray()).toEqual([5, 7, 9]);
    expect(a.sum(1).toArray()).toEqual([6, 15]);
    expect(a.sum(-1, true).shape).toEqual([2, 1]);
    expect(a.sum().item()).toBe(21);
    expect(a.mean().item()).toBeCloseTo(3.5, 5);
  });

  it("transposes 2-D tensors", () => {
    const a = Tensor.fromArray([1, 2, 3, 4, 5, 6], [2, 3]);
    expect(a.transpose().shape).toEqual([3, 2]);
    expect(a.transpose().toArray()).toEqual([1, 4, 2, 5, 3, 6]);
  });
});
