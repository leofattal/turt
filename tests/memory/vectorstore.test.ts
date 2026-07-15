import { describe, expect, it } from "vitest";
import { InMemoryVectorStore, cosineSimilarity } from "../../src/index.js";

describe("InMemoryVectorStore", () => {
  it("computes cosine similarity", () => {
    expect(cosineSimilarity([1, 0], [1, 0])).toBeCloseTo(1, 5);
    expect(cosineSimilarity([1, 0], [0, 1])).toBeCloseTo(0, 5);
    expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1, 5);
  });

  it("returns nearest records first", () => {
    const store = new InMemoryVectorStore();
    store.add({ id: "cat", text: "a cat", embedding: [1, 0, 0] });
    store.add({ id: "dog", text: "a dog", embedding: [0.9, 0.1, 0] });
    store.add({ id: "car", text: "a car", embedding: [0, 0, 1] });

    const results = store.search([1, 0, 0], 2);
    expect(results.map((r) => r.record.id)).toEqual(["cat", "dog"]);
    expect(results[0].score).toBeGreaterThan(results[1].score);
  });

  it("supports removal", () => {
    const store = new InMemoryVectorStore();
    store.add({ id: "x", text: "x", embedding: [1] });
    expect(store.size()).toBe(1);
    expect(store.remove("x")).toBe(true);
    expect(store.size()).toBe(0);
  });
});
