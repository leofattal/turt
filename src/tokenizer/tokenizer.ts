/**
 * Common tokenizer interface. BPE is implemented; WordPiece and a
 * SentencePiece-compatible adapter should conform to this same interface
 * (see PRD "Tokenization").
 */
export interface Tokenizer {
  readonly vocabSize: number;
  encode(text: string): number[];
  decode(ids: number[]): string;
}
