# Turt 🐢

A modular AI foundation platform built **from first principles** in TypeScript — tensor math, automatic differentiation, neural networks, optimizers, tokenization, training, inference backends, long-term memory, and tool use. No ML framework dependencies.

It includes a working **GPT** — a decoder-only transformer in the GPT-2 / nanoGPT shape — trained on 60 MB of Project Gutenberg text. Every layer, the autodiff that trains it, and the BPE tokenizer that feeds it are implemented here from scratch.

See [PRD.md](./PRD.md) for the product vision, [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) for the codebase map, and [docs/TRAINING.md](./docs/TRAINING.md) for the full training pipeline.

## Quickstart

```bash
pnpm install
pnpm test        # run the test suite
pnpm example     # autodiff, training, BPE, and memory in one script
pnpm build       # compile to dist/
```

## Train a language model

```bash
pnpm prepare-data                 # 73 Gutenberg books -> 60 MB -> 24.7M BPE tokens
pnpm train --steps 6500           # data-parallel across CPU cores
pnpm generate --prompt "It was a" # sample from the trained checkpoint
```

A 2.04M-parameter GPT trained this way for 12h on a 10-core M4 (CPU only) reaches a validation loss of **3.20**, down from the 6.93 uniform-prior baseline:

> It was a bright girl.
>
> “Why’s the barricade of your soul?” said Mr. Guppy; “but all him to-day, I have always been given; and they are silent, that they were very simply unable to get away on.”

Fluent English morphology, dialogue convention, and Victorian register — with the incoherent semantics you'd expect at 2M parameters. See [docs/TRAINING.md](./docs/TRAINING.md) for the full run and how the model is sized against the compute budget.

## Use it as a library

```ts
import { Tensor, Linear, Adam, Trainer, mulberry32 } from "turt";

// y = 2x + 1
const rng = mulberry32(7);
const xs = Array.from({ length: 64 }, () => rng() * 2 - 1);
const input = Tensor.fromArray(xs, [64, 1]);
const target = Tensor.fromArray(xs.map((x) => 2 * x + 1), [64, 1]);

const model = new Linear(1, 1, { rng });
const trainer = new Trainer(model, new Adam(model.parameters(), { lr: 0.05 }));
trainer.fit([{ input, target }], { epochs: 300 });

console.log(model.weight.data[0], model.bias!.data[0]); // ≈ 2, ≈ 1
```

## Project layout

| Directory       | Subsystem                                                            | Status |
| --------------- | -------------------------------------------------------------------- | ------ |
| `src/math`      | Tensors, broadcasting, reverse-mode autodiff, seeded RNG             | ✅ working core |
| `src/nn`        | Module system, Linear, activations, LayerNorm, losses                | ✅ working core |
| `src/models`    | GPT (CPU) + **GpuGPT** (GPU-resident training, 52× CPU)              | ✅ working core |
| `src/optim`     | SGD(+momentum), Adam/AdamW, RMSProp, LR schedulers, grad clipping    | ✅ working core |
| `src/tokenizer` | Byte-level BPE (train/encode/decode/serialize)                       | ✅ working core |
| `src/train`     | Mini-batch trainer with early-stop callback                          | ✅ minimal |
| `scripts/`      | Corpus prep, data-parallel training, checkpointing, generation       | ✅ working |
| `src/infer`     | Backend interface, CPU backend, **WebGPU backend + GPU-resident autograd** (validated on Metal) | ✅ working |
| `src/memory`    | Vector store with cosine semantic search                             | ✅ minimal |
| `src/tools`     | Tool interface + registry                                            | ✅ minimal |
| `src/agent`     | Coding-agent interfaces                                              | 🔲 interface only |

Every subsystem is independently replaceable — modules communicate only through the exported interfaces.

## Roadmap (from the PRD)

- **Math**: gradient checkpointing, WASM-SIMD kernels
- **NN**: CNN/RNN/LSTM/GRU, transformer *encoder*, rotary embeddings, MoE
- **Tokenization**: WordPiece, SentencePiece-compatible adapter, streaming
- **Training**: dataset streaming, multi-machine distribution, experiment tracking
- **Inference/GPU**: kernel fusion + half precision to widen the GPU lead, then cloud NVIDIA scale-out (the GPU-resident trainer is validated and 52× CPU — see [docs/GPU.md](./docs/GPU.md)); KV caching, streaming responses, quantization
- **Memory**: ranking, context compression, RAG assembly, pruning
- **Agent**: repository reading, planning, refactoring, review

## Development

- `pnpm test:watch` — watch mode
- `pnpm lint` — ESLint
- `pnpm format` — Prettier
