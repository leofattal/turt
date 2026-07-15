/**
 * Long-term memory: embedding storage with semantic (cosine) search.
 *
 * The in-memory store is the reference implementation; persistent or
 * approximate-nearest-neighbor stores should implement the same interface.
 * Ranking, context compression, RAG assembly, and pruning build on top of
 * this (see PRD "Long-Term Memory").
 */

export interface MemoryRecord {
  id: string;
  text: string;
  embedding: number[];
  metadata?: Record<string, unknown>;
}

export interface SearchResult {
  record: MemoryRecord;
  score: number;
}

export interface VectorStore {
  add(record: MemoryRecord): void;
  search(embedding: number[], topK: number): SearchResult[];
  remove(id: string): boolean;
  size(): number;
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) throw new Error("Embedding dimensions must match");
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

export class InMemoryVectorStore implements VectorStore {
  private records = new Map<string, MemoryRecord>();

  add(record: MemoryRecord): void {
    this.records.set(record.id, record);
  }

  search(embedding: number[], topK: number): SearchResult[] {
    const results: SearchResult[] = [];
    for (const record of this.records.values()) {
      results.push({ record, score: cosineSimilarity(embedding, record.embedding) });
    }
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, topK);
  }

  remove(id: string): boolean {
    return this.records.delete(id);
  }

  size(): number {
    return this.records.size;
  }
}
