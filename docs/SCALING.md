# Scaling Turt toward GPT-2 / GPT-3-class quality

This document is the honest answer to "make it like ChatGPT." It separates what
the local machine can do from what actually reaches GPT-2- or GPT-3-class
quality, and gives the turnkey path to the latter on a rented NVIDIA GPU.

## The one graph that governs everything

Language-model quality is set by **compute**, and compute is the product of three
things: model size (params `N`), tokens trained (`D`), and how efficiently the
hardware turns FLOPs into gradient steps. A transformer costs about `6 * N` FLOPs
per token (forward + backward), so a run costs `C ≈ 6 * N * D` FLOPs.

Two consequences drive every decision here:

1. **Chinchilla balance.** For a fixed compute budget `C`, loss is minimized when
   `D ≈ 20 * N` — roughly twenty training tokens per parameter. Smaller `D/N`
   under-trains a big model (GPT-3 itself ran at ~1.7 tokens/param, badly
   under-trained by this rule); larger `D/N` over-trains a small one (fine, just
   not compute-optimal).
2. **Hardware sets the ceiling.** The number of tokens you can process in a fixed
   wall-clock is `throughput (tok/s) * seconds`. That, through Chinchilla, caps
   the useful model size.

## Two ways to buy quality

Compute is the ceiling, but it is not the only lever — at a *fixed* compute budget
you still choose how well each parameter is spent. That is the architecture, and
it is the one lever that does not require a bigger GPU:

- **Architecture** (free, bounded): the `modern` shape — RMSNorm, SwiGLU, RoPE, no
  biases — measures 0.33 nats below the original `gpt2` shape at matched
  parameters, reaching `gpt2`'s 400-step loss in 200 steps. Worth taking, but it
  is a constant factor, not an order of magnitude. See
  [TRAINING.md](./TRAINING.md#architecture-arch-modern-default-vs-arch-gpt2).
- **Data quality** (free, underexploited): the Phi models' result — that a small
  model trained on dense, clean, pedagogical text beats a much larger one trained
  on scraped noise — applies directly here. This corpus is raw public-domain
  Gutenberg: archaic prose, no reasoning or instructional content. At ~10M params
  the model has capacity for grammar and local coherence and nothing to spare, so
  what it spends that capacity *on* is a real choice.
- **Compute** (expensive, unbounded): everything below.

The first two are worth exhausting before renting anything, but neither closes a
6-order-of-magnitude gap. Only compute does.

## Where the tiers land

| Tier            | Params | Chinchilla tokens (`20N`) | Compute `6ND` (FLOPs) |
| --------------- | ------ | ------------------------- | --------------------- |
| Turt local (this repo) | ~10–45M | ~0.2–0.9B | ~10^16–10^17 |
| GPT-2 small     | 124M   | ~2.5B                     | ~2 × 10^18            |
| GPT-2 large     | 774M   | ~15B                      | ~7 × 10^19            |
| GPT-3 (175B)    | 175B   | ~3.5T (ran on ~0.3T)      | ~3 × 10^23            |

The jump from Turt-local to GPT-3 is **~6–7 orders of magnitude of compute**. No
single consumer machine closes that; the honest path is (a) get the most out of
local hardware as a validated, correct pipeline, then (b) rent NVIDIA GPUs and
scale the *same code*.

## Local ceiling (Apple M4, WebGPU/Metal)

The whole training step runs GPU-resident (`src/models/gpu-gpt.ts` +
`src/infer/webgpu/autograd.ts`), validated bit-for-bit against the CPU engine.
Measured throughput and the model it justifies are recorded in
[TRAINING.md](./TRAINING.md); at the local rate a ~1–2 day run reaches the
10–45M-param tier — a >10× jump over the original 2M CPU model, but still two
tiers below GPT-2 small. This is the correctness-and-throughput floor the cloud
path builds on, not the destination.

## The cloud path (same code, bigger GPU)

WebGPU was chosen precisely so the kernels are portable: the identical WGSL that
runs on Metal here runs on cloud NVIDIA through Vulkan under Deno — no CUDA
rewrite. Step-by-step instructions — including a $0 route via Google Colab's
free T4 tier — are in [CLOUD.md](./CLOUD.md).

Concretely, to reach **GPT-2-small-class** quality:

1. Rent one modern data-center GPU (A100/H100 40–80GB). It delivers ~10^3× the
   sustained FLOP/s of the M4's WebGPU path, turning a 1–2 day local run into
   minutes and lifting the size ceiling by ~1000×.
2. Grow the corpus with the same `prepare-data.ts` sweep (it already scales to
   hundreds of MB; point it at more of Gutenberg, or add an open web/text
   dataset) so there are ~2.5B+ unique tokens for a 124M model.
3. Run the same `scripts/gpu-pretrain.ts` with `--layers 12 --heads 12 --embd
   768 --block 1024` and a batch sized to fill GPU memory. The supervisor
   `scripts/gpu-pretrain-loop.sh` auto-resumes from the last checkpoint across
   the multi-hour run.
4. Instruction-tune with `scripts/finetune-chat.ts --base <checkpoint>` on a
   richer conversational dataset than the synthetic one here.

To go further (GPT-2-large, then genuinely GPT-3-class) is the same recipe with
more GPUs and more data — at GPT-3 scale, a multi-node cluster and a curated
trillion-token corpus, which is a budget/logistics question, not a code one. The
point of Turt is that the pipeline from raw text to a chatting model is complete
and correct end to end; scale is the remaining variable.

## What "like ChatGPT" realistically means at each stop

- **Local (10–45M):** fluent English, correct grammar and dialogue form, coherent
  short spans; no reliable facts or reasoning. A convincing demo, not an
  assistant.
- **GPT-2 small (124M):** paragraph-level coherence, weak but present world
  knowledge, follows simple instructions after tuning. Feels like an early
  chatbot.
- **GPT-3 (175B):** the in-context learning and broad competence people mean by
  "ChatGPT." Requires the cluster-scale compute above.
