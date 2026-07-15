/**
 * Turt — a modular AI foundation platform built from first principles.
 *
 * Subsystems (each independently replaceable, per the PRD):
 * - math:      tensors, broadcasting, reverse-mode autodiff
 * - nn:        neural-network layers built on the math engine
 * - models:    GPT (decoder-only transformer, GPT-2 shape)
 * - optim:     optimizers, LR schedulers, gradient clipping
 * - tokenizer: byte-level BPE (WordPiece/SentencePiece planned)
 * - train:     mini-batch training loop
 * - infer:     compute-backend interface (CPU now; WebGPU/CUDA planned)
 * - memory:    vector store with semantic search
 * - tools:     extensible tool interface + registry
 * - agent:     coding-agent interfaces (planned)
 */

export * from "./math/index.js";
export * from "./nn/index.js";
export * from "./models/index.js";
export * from "./optim/index.js";
export * from "./tokenizer/index.js";
export * from "./train/index.js";
export * from "./infer/index.js";
export * from "./memory/index.js";
export * from "./tools/index.js";
export * from "./agent/index.js";
