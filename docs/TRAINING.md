# Training a GPT with Turt

This guide covers the full pipeline: corpus → tokenizer → training → generation. Everything runs on CPU, from scratch, with no ML framework.

## 1. Prepare the corpus

```bash
pnpm prepare-data --target-mb 60 --vocab 1024
```

Downloads public-domain books from Project Gutenberg, strips their license boilerplate, trains a byte-level BPE vocabulary, and writes the tokenized corpus.

| Output | Contents |
| --- | --- |
| `data/raw/<id>.txt` | Cached downloads — re-runs never re-fetch |
| `data/corpus.txt` | Cleaned, concatenated text |
| `data/tokenizer.json` | Learned BPE merges |
| `data/train.bin` / `val.bin` | Uint16 token ids, 90/10 split |
| `data/meta.json` | Vocab size, token counts, book list |

The default run yields **73 books, 60 MB, 24.7M tokens** at ~2.55 chars/token.

### Why 60 MB?

Compute, not disk, is the binding constraint. The engine sustains ~2.3 GFLOP/s per core (~7 GFLOP/s across the worker pool), and a GPT step costs roughly `6 × params` FLOPs per token. A 2M-parameter model therefore consumes ~27M tokens in a 12-hour run — barely more than one pass over this corpus. A larger corpus would be text the model never reads.

## 2. Train

```bash
pnpm train --steps 6500 --batch 32 --block 128 \
           --layer 6 --head 4 --embd 160 --lr 6e-4 --workers 8
```

Flags: `--steps --batch --block --layer --head --embd --lr --warmup --workers --eval-every --resume`.

- **Data-parallel across cores.** The batch is split into micro-batches; each worker thread runs forward/backward on its slice against a model replica, and the main thread averages the gradients and takes the optimizer step. This is mathematically identical to single-threaded training over the full batch — verified by the loss matching to 4 decimal places across 1, 4, and 8 workers — and gives a ~5× speedup on this 10-core machine.
- **Random-window sampling.** Batches are drawn from random offsets, so the corpus can exceed what one run traverses.
- **AdamW** (weight decay 0.1), gradient clipping at norm 1.0, linear warmup then cosine decay to 10% of peak LR.
- **Checkpoints** land in `checkpoints/`: `gpt.bin` (best validation loss) and `gpt-last.bin` (most recent, for `--resume`).
- Every `--eval-every` steps it prints validation loss and a sample so you can watch text quality improve.

### Sizing the model

Balancing the Chinchilla rule (`tokens ≈ 20 × params`) against a fixed compute budget `C` gives `params ≈ sqrt(C / 120)`. For a 12-hour run on this box that lands at ~1.5–2M parameters — hence the 6-layer, 160-dim default.

A useful sanity check: initial loss should be ≈ `ln(vocabSize)` (6.93 for a 1024-token vocab). A model that starts far from that is mis-initialized.

## Reference run

The default configuration above, trained on this corpus:

| | |
| --- | --- |
| Model | 2.04M params — 6 layers, 4 heads, 160 dim, 128-token context |
| Data | 73 Gutenberg books, 60 MB, 24.7M BPE tokens (vocab 1024) |
| Hardware | Apple M4, 10 cores, CPU only |
| Throughput | ~620 tok/s (8 data-parallel workers) |
| Wall clock | 12h26m, 6500 steps, 26.6M tokens |
| **Validation loss** | **6.93 → 3.20** (start = ln 1024, the uniform-prior baseline) |

Loss by step: 4.70 (250) → 3.90 (1k) → 3.55 (2k) → 3.39 (3k) → 3.31 (4k) → 3.24 (5k) → **3.20** (6.5k).

Sample at `--temperature 0.8 --top-k 40`, prompt `"It was a"`:

> It was a bright girl.
>
> “Why’s the barricade of your soul?” said Mr. Guppy; “but all him to-day, I have always been given; and they are silent, that they were very simply unable to get away on.”
>
> “I cannot imperceive you,” obser…

The model has learned English morphology, dialogue convention (matched curly quotes, `,” said X` attribution), and Victorian narrative register — it even reuses `Mr. Guppy`, a real character from *Bleak House*, in grammatically correct positions. Sentence-level semantics remain incoherent, which is the expected ceiling at 2M parameters: local fluency without long-range meaning. Reaching semantic coherence needs roughly two orders of magnitude more parameters and compute, which is a GPU-backend problem, not a data problem.

## 3. Generate

```bash
pnpm generate --prompt "It was a" --tokens 200 --temperature 0.8 --top-k 40
```

Loads the config from the checkpoint header, so no flags need to match the training run.

## Scaling beyond one box

The `Backend` interface in `src/infer` is the extension point for a WebGPU or CUDA backend. Model code is device-agnostic, so a GPU backend would drop in without touching `src/models/gpt.ts` — that is the path to training a model large enough to be genuinely useful rather than merely coherent.
