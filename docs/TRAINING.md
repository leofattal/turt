# Training a GPT with Turt

This guide covers the full pipeline: corpus → tokenizer → training → generation, from scratch, with no ML framework.

There are two training paths. The **CPU data-parallel trainer** (`pnpm train`, sections 1–3 below) is the original, capped at ~2M params by pure-JS throughput. The **GPU-resident trainer** (`pnpm gpu-pretrain`, section 4) keeps the whole step in GPU memory and lifts that ceiling ~50×, which is what the larger models here use. Both consume the same corpus and produce the same portable checkpoint format.

## 1. Prepare the corpus

```bash
pnpm prepare-data                      # defaults: --target-mb 250 --vocab 8192
pnpm prepare-data --target-mb 60 --vocab 1024   # the original small corpus
```

Downloads public-domain books from Project Gutenberg, strips their license boilerplate, trains a byte-level BPE vocabulary, and writes the tokenized corpus. It fetches a curated classics list first, then tops up to the target size with a deterministic sweep over Gutenberg ids, keeping only text that passes an English/quality filter.

| Output | Contents |
| --- | --- |
| `data/raw/<id>.txt` | Cached downloads — re-runs never re-fetch |
| `data/corpus.txt` | Cleaned, concatenated text |
| `data/tokenizer.json` | Learned BPE merges |
| `data/train.bin` / `val.bin` | Uint16 token ids, 90/10 split |
| `data/meta.json` | Vocab size, token counts, book list |

The default run yields **550 books, 250 MB, 77.7M tokens** at ~3.37 chars/token (vocab 8192). The original small preset yields 73 books, 60 MB, 24.7M tokens at ~2.55 chars/token (vocab 1024).

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

## 3. Train on the GPU (larger models)

The GPU-resident trainer runs the entire forward/backward/AdamW step in GPU memory
(only token ids in, scalar loss out), validated bit-for-bit against the CPU engine.
It needs [Deno](https://deno.com) on PATH for its native WebGPU (Metal on Apple
silicon, Vulkan on cloud NVIDIA — the same WGSL runs on both).

```bash
# One-shot:
pnpm gpu-pretrain --layers 6 --heads 6 --embd 384 --block 256 \
                  --batch 32 --steps 12000 --lr 6e-4 --out gpt-big.bin

# Or supervised (auto-resumes from the last checkpoint if the process dies —
# use this for multi-hour/day runs):
LAYERS=6 HEADS=6 EMBD=384 BLOCK=256 BATCH=32 STEPS=12000 LR=6e-4 \
  OUT=gpt-big.bin pnpm gpu-pretrain-loop
```

Model size comes from the flags (pretraining is where the architecture is chosen);
vocab and block are validated against the tokenizer/data. Checkpoints save every
`--eval-every` steps in the portable CPU format, so `generate`, `chat`, and
`finetune-chat` consume them unchanged.

### Architecture: `--arch modern` (default) vs `--arch gpt2`

`--arch` selects the transformer shape. The default `modern` is what separates a
2019 GPT-2 from a strong small model today — the same four changes Llama,
TinyLlama, and Qwen all converged on:

| | `gpt2` | `modern` |
| --- | --- | --- |
| Norm | LayerNorm (gain + bias) | **RMSNorm** (gain only) |
| Feed-forward | 4·c GELU | **SwiGLU** at 8/3·c |
| Positions | learned `wpe` table | **RoPE** (rotary, no parameters) |
| Biases | on every projection | **none** |

Why each earns its place at this scale:

- **SwiGLU** — the multiplicative gate expresses suppression a single GELU
  projection cannot. Sized at 8/3·c across three matrices so it has *identical*
  parameters and FLOPs to the two-matrix GELU block: the gain is free, not bought
  with width.
- **RoPE** — encodes *relative* position directly in the q·k dot product, at zero
  parameters: a whole `blockSize × nEmbd` table disappears. The rotation is
  defined at any position, so extending context later is a flag change rather
  than a retrained table — though `forward()` still enforces `blockSize` today.
- **RMSNorm** — matches LayerNorm without centering; drops a reduction and a
  parameter vector per norm.
- **No biases** — free parameters at no measured quality cost.

**Measured, not assumed.** Both shapes pretrained on this corpus at an identical
config (4L/4H/256E, block 128, batch 32, 400 steps ≈ 1.6M tokens, same seed and
data), vocab 8192:

| arch | params | val loss @100 | @200 | @300 | @400 |
| --- | --- | --- | --- | --- | --- |
| `gpt2` | 5.29M | 6.5397 | 6.0487 | 5.8829 | 5.8153 |
| `modern` | 5.31M | **6.4099** | **5.7805** | **5.5672** | **5.4816** |

`modern` ends **0.33 nats lower at matched parameters** (+0.4% params), and the gap
widens with training rather than closing. The sharper way to read it: `modern` beat
`gpt2`'s *final* 400-step loss by step 200 — the same quality for half the compute.

That dwarfs its cost. In a controlled benchmark (`pnpm gpu-gpt`) `modern` runs at
~0.97x `gpt2`'s throughput — RoPE and SwiGLU's third matmul slightly outweigh the
six dispatches saved by dropping biases — and in the runs above that 3% vanished
into wall-clock noise (10m vs 11m). Since the engine is dispatch-bound, that
overhead is op count, not arithmetic.

Both architectures are validated gradient-for-gradient against the CPU engine by
`pnpm gpu-gpt`. `arch` lives in the checkpoint, so `--resume` always adopts the
checkpoint's own shape and ignores the flag; checkpoints written before `arch`
existed load as `gpt2` and keep working.

### Sizing on this hardware — the real constraint

The Deno/WebGPU-on-Metal path is **dispatch-bound**: each tensor op is a separate
GPU kernel launch, so throughput is set by op count, not raw FLOPs. Measured on the
M4 at vocab 8192 / block 256: a 29.5M-param model (8L/512) runs ~500 tok/s; the
13.9M model below runs faster because it has fewer layers (fewer dispatches). At
~500–800 tok/s a 1–2 day laptop run processes ~40–140M tokens, so by the Chinchilla
rule (`tokens ≈ 20 × params`) the **compute-optimal model is only ~8–15M params**.
A bigger model would be under-trained and finish at *higher* loss — so "train it
well" here means a ~14M model trained near-optimally, not the largest that fits.
The genuine jump to GPT-2/GPT-3-class quality is a hardware move to a cloud GPU,
laid out in [SCALING.md](./SCALING.md).

### GPU reference run

| | |
| --- | --- |
| Model | 13.9M params — 6 layers, 6 heads, 384 dim, 256-token context |
| Data | 550 Gutenberg books, 250 MB, 77.7M BPE tokens (vocab 8192) |
| Hardware | Apple M4, WebGPU/Metal (Deno) |
| Schedule | batch 32×256, 12000 steps ≈ 98M tokens, cosine LR peak 6e-4 |
| **Validation loss** | _run in progress — updated on completion_ |

A useful sanity check: initial loss should be ≈ `ln(vocabSize)` = 8.98 for the
8192-token vocab; the run starts there and decays.

## 4. Generate

```bash
pnpm generate --ckpt checkpoints/gpt-big.bin --prompt "It was a" --tokens 200 --temperature 0.8 --top-k 40
```

Loads the config from the checkpoint header, so no flags need to match the training run.

## Scaling beyond one box

The WebGPU backend in `src/infer/webgpu` already lifts the CPU ceiling ~50× (section 3), and its WGSL runs unchanged on cloud NVIDIA via Vulkan. The remaining path to a model large enough to be genuinely useful rather than merely coherent — GPT-2- then GPT-3-class — is a hardware move to rented GPUs, with the same code and a larger corpus. The compute math and the turnkey steps are in [SCALING.md](./SCALING.md).
