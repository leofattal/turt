# Training Turt on a cloud GPU (free and paid)

This is the practical companion to [SCALING.md](./SCALING.md): exact steps to
run the existing pipeline on an NVIDIA GPU you don't own. The code needs no
changes — the WGSL kernels that run on the Mac's Metal run on NVIDIA through
Vulkan under Deno. The only variables are which GPU, and who pays.

| Path | Cost | GPU | What it realistically buys |
| ---- | ---- | --- | -------------------------- |
| Google Colab (free tier) | $0 | T4 16GB | ~30–60M-param model; validates the whole NVIDIA path |
| Kaggle Notebooks | $0 (30 GPU-h/week) | T4 ×2 / P100 | same tier as Colab, more predictable quota |
| RunPod / Vast.ai rental | ~$0.3–2/hr (~$20–100 total) | RTX 4090 / A100 80GB | GPT-2-small-class (~100–124M params) |

A free T4 is only ~2–3× the M4's sustained f32 throughput — it does **not**
close the gap to GPT-2. What it does do, for $0: prove the code runs on
NVIDIA/Vulkan, measure real tok/s (which turns the paid-path cost from a guess
into arithmetic), and train a model 2–4× bigger than the local ceiling.

## What every path has in common

1. **Corpus.** `scripts/prepare-data-big.ts` streams C4 (cleaned Common Crawl,
   plain HTTPS, no auth) into `data-big/{tokenizer.json,train.bin,val.bin,meta.json}`.
   Default 8GB of text ≈ 1.9B tokens — sized for GPT-2-small and, deliberately,
   for the ~4GB typed-array ceiling on `train.bin`. Free tier wants less:
   `--target-gb 3` (~0.7B tokens) matches a 33M model.
2. **Trainer.** `scripts/gpu-pretrain.ts --data data-big` plus the shape flags.
   The supervisor `scripts/gpu-pretrain-loop.sh` auto-resumes from the last
   checkpoint after any crash, driver hiccup, or session preemption — this is
   what makes free tiers (which kill sessions) workable at all.
3. **Benchmark before committing.** Run a few hundred steps, read the printed
   `tok/s`, then: `hours = target_tokens / tokPerSec / 3600`. Decide with that
   number, not estimates.

Checkpoints save every 500 steps (`--eval-every`), so a preemption costs at
most 500 steps of progress.

## Free path: Google Colab

Caveats to know going in: sessions last up to ~12h and can be preempted
earlier; the VM disk is wiped on disconnect (so checkpoints must live on Google
Drive); free-tier availability varies by time of day. The resume loop makes
this survivable: when a session dies, rerun the cells and it continues.

Open a new notebook at colab.research.google.com, set **Runtime → Change
runtime type → T4 GPU**, then run these cells.

**Cell 1 — mount Drive (checkpoints must outlive the VM):**

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 2 — clone + setup** (`scripts/colab-setup.sh` installs Vulkan and the
NVIDIA ICD, Deno, Node 22, pnpm, and project deps; symlinks `checkpoints/`
into Drive; and validates the GPU kernels bit-for-bit against the CPU engine —
it exits loudly if Vulkan only sees `llvmpipe`, i.e. CPU):

```bash
!git clone https://github.com/leofattal/turt.git /content/turt
!bash /content/turt/scripts/colab-setup.sh
```

**Cell 3 — benchmark (~10 min on top of the one-time ~1h corpus build):**

```bash
!STEPS=100 bash /content/turt/scripts/colab-train.sh
```

`colab-train.sh` restores the corpus from a Drive tarball — or on the very
first run builds 3GB of C4 (~0.7B tokens) and stashes the tarball — then
trains. `STEPS=100` makes it stop after 100 steps: read the printed `tok/s`,
then `0.7e9 / tokPerSec / 3600` is the total GPU-hours for the full run.
If `nvidia-smi` shows plenty of free memory, benchmark again with
`STEPS=200 BATCH=16` — bigger batch is almost always more tok/s (at batch 16,
halve STEPS to keep the token budget).

**Cell 4 — the real run (resumes from the checkpoint; rerun after any disconnect):**

```bash
!bash /content/turt/scripts/colab-train.sh
```

Defaults: 33M params (`LAYERS=8 HEADS=8 EMBD=512 BLOCK=512 BATCH=8`),
`STEPS=160000` ≈ 0.66B tokens, output `turt-c4-33m.bin`. All env-overridable.
Because `checkpoints/` is a symlink into Drive, every 500-step checkpoint is
safe. When a session dies: reopen, rerun cells 1, 2, and 4 — it resumes where
it stopped (the 100 benchmark steps count toward the run, too).

A modest step up that still fits a T4: `LAYERS=10 HEADS=10 EMBD=640` (~60M
params, ~1.2B-token Chinchilla target — likely a multi-week free-tier project;
benchmark first and do the arithmetic before committing to it).

## Free path: Kaggle

Same recipe, different wrapper: kaggle.com → New Notebook → Settings →
Accelerator **GPU T4 ×2**, and turn **Internet on**. You get ~30 GPU-hours per
week (shown in the sidebar), sessions up to 12h, and `/kaggle/working` persists
as a versioned output (20GB) — use it for `checkpoints/` instead of the Drive
symlink. The setup cells are identical from Cell 2 onward (drop the Drive
mount, clone into `/kaggle/working/turt`). The weekly quota makes multi-week
runs slower but the accounting more predictable than Colab.

## Paid path: rented GPU (GPT-2-small-class)

When there's a small budget (~$20–100 total), the same steps move to a rented
box and the size ceiling lifts to GPT-2-small-class. Prices as of mid-2026:
A100 80GB ≈ $1.1–2/hr (RunPod, Lambda, Spheron), H100 ≈ $1.5–2.5/hr, RTX 4090
≈ $0.31–0.70/hr (Vast.ai/RunPod marketplace).

Recommendation: **A100 80GB**. The 4090 is cheaper per FLOP but its 24GB caps
the batch size at block 1024, and this autograd keeps the whole graph resident;
the A100's 80GB is what lets the batch (and therefore tok/s) grow.

1. Create the instance: RunPod → Deploy → A100 80GB, any Ubuntu + CUDA base
   template (only the driver matters), ≥60GB disk. Connect via their web
   terminal or SSH.
2. Setup is Colab Cell 2 minus Drive/ICD-hack (real driver installs include the
   ICD; if `vulkaninfo --summary` doesn't show the A100, apply the same ICD
   block): install `libvulkan1 vulkan-tools`, Deno, Node 22, pnpm, clone, `pnpm install`.
3. Validate: `pnpm gpu-gpt`.
4. Corpus at full size (fast datacenter network): `pnpm prepare-data-big` (8GB
   of text, ~1.9B tokens, ~1–2h dominated by single-threaded BPE encoding).
5. Benchmark the GPT-2 shape, starting at `--batch 16` and raising it until
   `nvidia-smi` shows memory nearly full:
   `pnpm gpu-pretrain --data data-big --layers 12 --heads 12 --embd 768 --block 1024 --batch 16 --steps 100 --out bench.bin`
6. The run (in `tmux`, so SSH drops don't matter). `STEPS` = target tokens /
   (batch × 1024); for 1.9B tokens at batch 16 that is `STEPS=116000`:

   ```bash
   tmux new -s train
   LAYERS=12 HEADS=12 EMBD=768 BLOCK=1024 BATCH=16 STEPS=116000 \
     DATA=data-big OUT=gpt2-small.bin bash scripts/gpu-pretrain-loop.sh 2>&1 | tee train.log
   ```

   This shape is ~98M params with the 16k vocab (GPT-2's 124M counted a 50k
   vocab's 38M embedding; compute-wise this is the same class).
7. Bring the model home, then **stop the pod** — billing is per-hour-alive:

   ```bash
   scp <pod>:/root/turt/checkpoints/gpt2-small.bin checkpoints/
   scp <pod>:/root/turt/data-big/tokenizer.json data/tokenizer-16384.json
   ```

   The tokenizer copy uses the `tokenizer-<vocab>.json` convention that
   `loadMatchingTokenizer` already resolves, so `pnpm chat --model
   gpt2-small.bin` finds it by the checkpoint's vocab size.

Cost sanity check with a real number: if the benchmark shows e.g. 60k tok/s,
the 1.9B-token run is ~9 hours ≈ **$13–18** on an A100. If it shows 15k tok/s
(fp32-bound kernels), it's ~35 hours ≈ **$40–70** — still dinner money, but
worth knowing before, not after. That number is exactly what the free Colab
benchmark already told you, scaled by the A100/T4 ratio you'll measure.

## After pretraining

Chat-finetuning the C4 model needs the chat dataset re-encoded with the
`data-big` tokenizer (today `prepare-chat.ts` assumes `data/`) — a small
adaptation to make once a pretrained checkpoint exists. Base-model quality is
visible immediately via `pnpm generate` / the sample lines the trainer prints
at every eval.
