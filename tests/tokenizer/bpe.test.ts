import { describe, expect, it } from "vitest";
import { BPETokenizer } from "../../src/index.js";

const CORPUS =
  "the quick brown fox jumps over the lazy dog. " +
  "the quick brown fox jumps over the lazy dog. " +
  "the lazy dog sleeps while the quick fox runs. ";

describe("BPETokenizer", () => {
  it("grows the vocabulary through merges", () => {
    const tok = new BPETokenizer();
    expect(tok.vocabSize).toBe(256);
    tok.train(CORPUS, 300);
    expect(tok.vocabSize).toBeGreaterThan(256);
    expect(tok.vocabSize).toBeLessThanOrEqual(300);
  });

  it("round-trips text exactly, including unseen unicode", () => {
    const tok = new BPETokenizer();
    tok.train(CORPUS, 300);
    for (const text of [CORPUS, "the quick fox", "völlig neu — 日本語 🐢", ""]) {
      expect(tok.decode(tok.encode(text))).toBe(text);
    }
  });

  it("compresses trained text below raw byte length", () => {
    const tok = new BPETokenizer();
    tok.train(CORPUS, 320);
    const encoded = tok.encode("the quick brown fox jumps over the lazy dog.");
    const rawBytes = new TextEncoder().encode("the quick brown fox jumps over the lazy dog.").length;
    expect(encoded.length).toBeLessThan(rawBytes);
  });

  it("serializes and restores merges", () => {
    const tok = new BPETokenizer();
    tok.train(CORPUS, 300);
    const restored = BPETokenizer.fromJSON(tok.toJSON());
    expect(restored.vocabSize).toBe(tok.vocabSize);
    expect(restored.encode(CORPUS)).toEqual(tok.encode(CORPUS));
  });
});
