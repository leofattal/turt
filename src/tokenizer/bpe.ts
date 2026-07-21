import type { Tokenizer } from "./tokenizer.js";

/**
 * Byte-level byte-pair encoding, GPT-2 style.
 *
 * The base vocabulary is the 256 byte values; training greedily merges the
 * most frequent adjacent pair into a new token id (256, 257, ...). Any
 * unicode text round-trips exactly, since unmergeable sequences fall back to
 * raw bytes.
 *
 * Text is pre-tokenized into word-like chunks before BPE, and merges never
 * cross a chunk boundary. This is what makes the tokenizer usable on
 * multi-megabyte corpora:
 * - training counts pairs over *unique* chunks weighted by frequency, so cost
 *   scales with vocabulary size rather than corpus size;
 * - encoding memoizes each chunk's token ids, so repeated words are free.
 */

/**
 * Splits text into contractions, letter runs, digit runs, punctuation runs,
 * and whitespace runs (each optionally preceded by a single space).
 */
const CHUNK_PATTERN = /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+/gu;

/** Packs an adjacent (left, right) token pair into one number key. */
const PAIR_STRIDE = 1_000_000;
const packPair = (left: number, right: number): number => left * PAIR_STRIDE + right;

/**
 * Splits into chunks with guaranteed full coverage: any character the pattern
 * fails to match is emitted as its own chunk, so concatenating chunks always
 * reproduces the input exactly.
 */
function chunkText(text: string): string[] {
  const chunks: string[] = [];
  let cursor = 0;
  CHUNK_PATTERN.lastIndex = 0;
  let match: RegExpExecArray | null;
  while ((match = CHUNK_PATTERN.exec(text)) !== null) {
    if (match.index > cursor) chunks.push(text.slice(cursor, match.index));
    chunks.push(match[0]);
    cursor = match.index + match[0].length;
  }
  if (cursor < text.length) chunks.push(text.slice(cursor));
  return chunks;
}

export class BPETokenizer implements Tokenizer {
  /** packed pair -> merged token id. Lower ids were learned earlier (lower rank). */
  private merges = new Map<number, number>();
  /** merged id - 256 -> its [left, right] parts. */
  private parts: Array<[number, number]> = [];
  /** chunk text -> encoded ids, populated lazily during encode. */
  private cache = new Map<string, number[]>();

  get vocabSize(): number {
    return 256 + this.parts.length;
  }

  /** Learns merges from `text` until the vocabulary reaches `vocabSize`. */
  train(text: string, vocabSize: number, onProgress?: (vocab: number, target: number) => void): void {
    if (vocabSize < 256) throw new Error("vocabSize must be at least 256 (byte-level base)");
    this.cache.clear();

    // Count unique chunks; BPE then operates on the unique set, weighted.
    const frequencies = new Map<string, number>();
    for (const chunk of chunkText(text)) {
      frequencies.set(chunk, (frequencies.get(chunk) ?? 0) + 1);
    }
    const encoder = new TextEncoder();
    const words: number[][] = [];
    const counts: number[] = [];
    for (const [chunk, count] of frequencies) {
      words.push(Array.from(encoder.encode(chunk)));
      counts.push(count);
    }

    while (this.vocabSize < vocabSize) {
      const pairCounts = new Map<number, number>();
      for (let w = 0; w < words.length; w++) {
        const ids = words[w];
        const count = counts[w];
        for (let i = 0; i < ids.length - 1; i++) {
          const key = packPair(ids[i], ids[i + 1]);
          pairCounts.set(key, (pairCounts.get(key) ?? 0) + count);
        }
      }

      let bestKey = -1;
      let bestCount = 1; // require a pair seen at least twice
      for (const [key, count] of pairCounts) {
        if (count > bestCount) {
          bestCount = count;
          bestKey = key;
        }
      }
      if (bestKey < 0) break;

      const left = Math.floor(bestKey / PAIR_STRIDE);
      const right = bestKey % PAIR_STRIDE;
      const newId = this.vocabSize;
      this.merges.set(bestKey, newId);
      this.parts.push([left, right]);
      for (let w = 0; w < words.length; w++) {
        if (words[w].length >= 2) words[w] = mergePair(words[w], left, right, newId);
      }
      onProgress?.(this.vocabSize, vocabSize);
    }
  }

  encode(text: string): number[] {
    const out: number[] = [];
    for (const chunk of chunkText(text)) {
      const cached = this.cache.get(chunk);
      if (cached) {
        for (const id of cached) out.push(id);
        continue;
      }
      const ids = this.encodeChunk(chunk);
      // Bound the memo: on corpus-scale input (billions of chars of web text)
      // unique chunks otherwise grow without limit and exhaust the heap.
      if (this.cache.size >= 1_000_000) this.cache.clear();
      this.cache.set(chunk, ids);
      for (const id of ids) out.push(id);
    }
    return out;
  }

  /** Applies merges to one chunk, always taking the lowest-rank applicable merge. */
  private encodeChunk(chunk: string): number[] {
    let ids: number[] = Array.from(new TextEncoder().encode(chunk));
    while (ids.length >= 2) {
      let bestId = Infinity;
      let bestLeft = -1;
      let bestRight = -1;
      for (let i = 0; i < ids.length - 1; i++) {
        const id = this.merges.get(packPair(ids[i], ids[i + 1]));
        if (id !== undefined && id < bestId) {
          bestId = id;
          bestLeft = ids[i];
          bestRight = ids[i + 1];
        }
      }
      if (bestLeft < 0) break;
      ids = mergePair(ids, bestLeft, bestRight, bestId);
    }
    return ids;
  }

  decode(ids: ArrayLike<number>): string {
    const bytes: number[] = [];
    const expand = (id: number): void => {
      if (id < 256) {
        bytes.push(id);
        return;
      }
      const pair = this.parts[id - 256];
      if (!pair) throw new Error(`Unknown token id ${id}`);
      expand(pair[0]);
      expand(pair[1]);
    };
    for (let i = 0; i < ids.length; i++) expand(ids[i]);
    // Partial multi-byte sequences can appear when decoding a truncated
    // generation; the decoder replaces them rather than throwing.
    return new TextDecoder().decode(new Uint8Array(bytes));
  }

  /** Serializable view of the learned merges (for checkpointing). */
  toJSON(): { parts: Array<[number, number]> } {
    return { parts: [...this.parts] };
  }

  static fromJSON(json: { parts: Array<[number, number]> }): BPETokenizer {
    const tok = new BPETokenizer();
    for (const [left, right] of json.parts) {
      const newId = tok.vocabSize;
      tok.merges.set(packPair(left, right), newId);
      tok.parts.push([left, right]);
    }
    return tok;
  }
}

function mergePair(ids: number[], left: number, right: number, newId: number): number[] {
  const out: number[] = [];
  let i = 0;
  while (i < ids.length) {
    if (i < ids.length - 1 && ids[i] === left && ids[i + 1] === right) {
      out.push(newId);
      i += 2;
    } else {
      out.push(ids[i]);
      i += 1;
    }
  }
  return out;
}
