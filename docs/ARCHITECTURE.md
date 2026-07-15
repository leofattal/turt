# Turt Architecture

Turt is organized as one TypeScript package with strictly layered subsystems. Lower layers never import from higher layers, and each subsystem is replaceable behind its exported interfaces.

```
agent  ──▶ tools, memory, infer, nn
scripts ─▶ models, optim, tokenizer, math   (corpus prep, training, generation)
models ──▶ nn, math       (GPT)
train  ──▶ nn, optim, math
infer  ──▶ math           (backend interface; CPU now, WebGPU/CUDA later)
nn     ──▶ math
optim  ──▶ math
tokenizer, memory, tools  (standalone)
math                      (foundation: no internal dependencies)
```

## Math engine (`src/math`)

- **`shape.ts`** — shape/stride utilities and NumPy-style broadcasting (`broadcastShapes`, `broadcastIndexer`).
- **`random.ts`** — seedable PRNG (mulberry32) + Box–Muller gaussian for reproducible init.
- **`tensor.ts`** — `Tensor`: dense row-major Float32 storage plus reverse-mode autodiff.
  - Every differentiable op attaches a `GradCtx { parents, backward }` to its output; `backward()` topologically sorts the graph and accumulates into `grad` buffers.
  - Element-wise and reduction ops are funneled through three helpers — `unary`, `binary` (broadcasting), and the reductions — so faster kernels (WASM-SIMD, GPU) can be swapped in without touching graph mechanics.
  - Transformer-specific ops carry hand-written backward passes: `gatherRows` (embedding lookup, scatter-add backward), `permute`, `bmm` (batched matmul), `softmax`, `maskedFillCausal`, `gelu`, and `crossEntropyLogits`. Cross-entropy is **fused** with softmax so probabilities never enter the graph and the backward pass reduces to the exact `(softmax − onehot) / n`.
  - Planned: gradient checkpointing (recompute-instead-of-store between checkpoints).

## Neural networks (`src/nn`)

`Module` is the composition primitive; `parameters()` auto-discovers trainable tensors by walking fields. Layers are built from primitive tensor ops so no layer writes a manual backward pass. Implemented: `Linear`, `ReLU`/`Tanh`/`Sigmoid`, `LayerNorm`, `Sequential`, `mseLoss`. Planned per the PRD: CNN/RNN/LSTM/GRU, transformer *encoder*, rotary embeddings, MoE.

## Models (`src/models`)

`GPT` is a decoder-only transformer in the GPT-2 shape, assembled entirely from the ops above:

```
idx -> token embedding + learned positional embedding
    -> nLayer x [ x + attn(ln1(x)) ; x + mlp(ln2(x)) ]   (pre-LayerNorm)
    -> final LayerNorm
    -> lm_head (weights tied to the token embedding)
```

Attention reshapes `[batch, time, channels]` into per-head `[batch*heads, time, headDim]` via `permute` + `reshape`, scores with `bmm`, applies the causal mask, and softmaxes — so autodiff supplies the entire backward pass. The output head shares weights with the token embedding (GPT-2 weight tying); gradients from both uses accumulate into the same tensor. `generate()` samples autoregressively with temperature and top-k, cropping context to `blockSize`.

Dropout is deliberately omitted: at a scale trainable on CPU the model is data-bound, not overfitting, so skipping it buys compute.

## Optimizers (`src/optim`)

`Optimizer` holds params and per-param state maps. Implemented: `SGD` (+momentum), `Adam`/`AdamW` (decoupled weight decay), `RMSProp`, `StepLR`/`CosineAnnealingLR` schedulers, `clipGradNorm`. Mixed precision awaits a second dtype in the math engine.

## Tokenization (`src/tokenizer`)

`Tokenizer` interface (`vocabSize`, `encode`, `decode`); `BPETokenizer` is byte-level BPE with exact round-tripping and JSON serialization for checkpoints. WordPiece and a SentencePiece-compatible adapter should implement the same interface.

## Training (`src/train`, `scripts/`)

`Trainer.fit` runs the canonical loop (zeroGrad → forward → loss → backward → step) with an epoch callback that doubles as early stopping — enough for small supervised models.

Language-model training lives in `scripts/` because it needs more: `prepare-data.ts` (corpus → BPE → token binaries), `train-gpt.ts` (the loop, LR schedule, validation, sampling), `train-worker.ts` (a model replica per core), `params.ts` (flat parameter/gradient views for crossing the thread boundary), and `checkpoint.ts` (a JSON header plus raw Float32 blocks, shape-verified on load).

**Data parallelism** is the one non-obvious piece. The batch is sharded across worker threads; each worker runs forward/backward on its slice against a replica and transfers gradients back; the main thread averages them (weighted by row count, recovering the exact full-batch mean) and takes the optimizer step. Because the reduction is exact, training is mathematically identical to a single-threaded run — the loss matches to four decimals across 1, 4, and 8 workers — while running ~5× faster on this 10-core machine. See [TRAINING.md](./TRAINING.md).

## Inference (`src/infer`)

`Backend` is the device-dispatch interface (`isAvailable`, op methods); `CpuBackend` delegates to the math engine. `GpuBackend` (`src/infer/webgpu`) implements the same interface with WebGPU: a tiled WGSL matmul kernel driving both `matmul` and batched `bmm`. The compute core (`GpuCompute`) is framework-agnostic — it operates on `Float32Array` and depends only on the standard WebGPU API — so it runs on Apple Metal, Vulkan/NVIDIA, or in a browser from one source. It is validated bit-for-bit against the CPU engine and reaches 52× CPU throughput on large matmuls (see [GPU.md](./GPU.md)). The generation loop, batching, streaming, KV caching, and quantization hooks build on this seam, keeping model code device-agnostic.

## Long-term memory (`src/memory`)

`VectorStore` interface with `InMemoryVectorStore` (cosine similarity, top-K). Ranking, context compression, RAG assembly, and pruning compose on top.

## Tool use (`src/tools`) and agent (`src/agent`)

`Tool` is a named, described, async `run(input)`; `ToolRegistry` handles registration and dispatch. The agent layer defines `Agent`/`AgentStep` and will compose model + memory + tools into plan/act loops.

## Testing

Vitest under `tests/`, mirroring `src/`. The autograd suite verifies analytic gradients against central-difference numeric gradients (including a diamond-graph case), and the NN suite trains real models end-to-end as integration tests.
