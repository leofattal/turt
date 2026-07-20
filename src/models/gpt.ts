/**
 * GPT — a decoder-only transformer built entirely on Turt's tensor engine.
 * Every layer composes primitive differentiable ops, so autodiff supplies the
 * whole backward pass.
 *
 * Two architectures share this file, selected by `GPTConfig.arch`:
 *
 *   "gpt2"   — the original 2019 shape (nanoGPT's): LayerNorm, GELU MLP,
 *              a learned position table, biases on every projection.
 *   "modern" — the shape every strong small model converged on (Llama,
 *              TinyLlama, Qwen): RMSNorm, SwiGLU, RoPE, no biases.
 *
 * Both are pre-norm with a weight-tied head:
 *
 *   idx -> token embedding (+ learned positional embedding, "gpt2" only)
 *       -> nLayer x [ x + attn(norm(x)) ; x + mlp(norm(x)) ]
 *       -> final norm
 *       -> lm_head (weights tied to the token embedding)
 *
 * "modern" is the default because it measures better: on this corpus at a matched
 * 5.3M params it ends 0.33 nats below "gpt2", reaching "gpt2"'s 400-step loss by
 * step 200 (numbers and method in docs/TRAINING.md). It costs ~3% throughput —
 * dropping the biases removes six kernel dispatches per block, but RoPE and
 * SwiGLU's third matmul spend that back. Since the engine is dispatch-bound
 * rather than FLOP-bound (docs/SCALING.md), that 3% is op count, not arithmetic.
 *
 * Dropout is omitted: at the scale trainable here the model is heavily
 * data-bound rather than overfitting, and skipping it buys compute.
 */

import { Tensor } from "../math/tensor.js";
import { type Rng, defaultRng } from "../math/random.js";
import { Module } from "../nn/module.js";
import { Linear } from "../nn/linear.js";
import { LayerNorm } from "../nn/layernorm.js";
import { RMSNorm } from "../nn/rmsnorm.js";

export type Arch = "gpt2" | "modern";

export interface GPTConfig {
  vocabSize: number;
  blockSize: number;
  nLayer: number;
  nHead: number;
  nEmbd: number;
  /**
   * Defaults to "gpt2" when absent, so checkpoints written before this field
   * existed keep loading with the architecture they were trained as.
   */
  arch?: Arch;
}

/** A norm layer; both variants take a [.., nEmbd] tensor and return the same shape. */
type Norm = LayerNorm | RMSNorm;

/** GPT-2 initializes most weights from N(0, 0.02). */
const INIT_STD = 0.02;

export const archOf = (config: GPTConfig): Arch => config.arch ?? "gpt2";

/**
 * Hidden width of the feed-forward block.
 *
 * GELU's MLP is two matrices of 4*nEmbd. SwiGLU needs three (gate, up, down),
 * so 8/3 * nEmbd keeps the parameter count identical to the GELU block rather
 * than quietly inflating the model by 50%. Rounded to a multiple of 64 to keep
 * the matmul tiles aligned.
 */
export function mlpHidden(config: GPTConfig): number {
  if (archOf(config) === "gpt2") return 4 * config.nEmbd;
  return Math.max(64, Math.round((8 * config.nEmbd) / 3 / 64) * 64);
}

const makeNorm = (config: GPTConfig, dim: number): Norm =>
  archOf(config) === "modern" ? new RMSNorm(dim) : new LayerNorm(dim);

/** Multi-head causal self-attention over a [batch, time, channels] input. */
class CausalSelfAttention extends Module {
  private readonly q: Linear;
  private readonly k: Linear;
  private readonly v: Linear;
  private readonly proj: Linear;
  private readonly nHead: number;
  private readonly headDim: number;
  private readonly useRope: boolean;

  constructor(config: GPTConfig, rng: Rng) {
    super();
    const { nEmbd, nHead, nLayer } = config;
    if (nEmbd % nHead !== 0) {
      throw new Error(`nEmbd (${nEmbd}) must be divisible by nHead (${nHead})`);
    }
    this.nHead = nHead;
    this.headDim = nEmbd / nHead;
    this.useRope = archOf(config) === "modern";
    const bias = archOf(config) === "gpt2";
    this.q = new Linear(nEmbd, nEmbd, { rng, std: INIT_STD, bias });
    this.k = new Linear(nEmbd, nEmbd, { rng, std: INIT_STD, bias });
    this.v = new Linear(nEmbd, nEmbd, { rng, std: INIT_STD, bias });
    // Residual projections are down-scaled so the residual stream variance
    // stays ~constant with depth (GPT-2's 1/sqrt(2 * nLayer) rule).
    this.proj = new Linear(nEmbd, nEmbd, { rng, std: INIT_STD / Math.sqrt(2 * nLayer), bias });
  }

  /** Splits [b*t, c] into per-head [b*h, t, headDim]. */
  private splitHeads(x: Tensor, b: number, t: number): Tensor {
    return x
      .reshape([b, t, this.nHead, this.headDim])
      .permute([0, 2, 1, 3])
      .reshape([b * this.nHead, t, this.headDim]);
  }

  forward(x: Tensor): Tensor {
    const [b, t, c] = x.shape;
    const flat = x.reshape([b * t, c]);

    let q = this.splitHeads(this.q.forward(flat), b, t);
    let k = this.splitHeads(this.k.forward(flat), b, t);
    const v = this.splitHeads(this.v.forward(flat), b, t);

    // Position enters here rather than at the embedding: rotating q and k makes
    // their dot product a function of relative distance.
    if (this.useRope) {
      q = q.rope();
      k = k.rope();
    }

    // Scaled dot-product attention with a causal mask.
    const scores = q
      .bmm(k.permute([0, 2, 1]))
      .mulScalar(1 / Math.sqrt(this.headDim))
      .maskedFillCausal()
      .softmax();

    const attended = scores
      .bmm(v) // [b*h, t, headDim]
      .reshape([b, this.nHead, t, this.headDim])
      .permute([0, 2, 1, 3]) // back to [b, t, h, headDim]
      .reshape([b * t, c]);

    return this.proj.forward(attended).reshape([b, t, c]);
  }
}

/** GPT-2's position-wise feed-forward network: c -> 4c -> GELU -> c. */
class GeluMLP extends Module {
  private readonly fc: Linear;
  private readonly proj: Linear;

  constructor(config: GPTConfig, rng: Rng) {
    super();
    const { nEmbd, nLayer } = config;
    const hidden = mlpHidden(config);
    this.fc = new Linear(nEmbd, hidden, { rng, std: INIT_STD });
    this.proj = new Linear(hidden, nEmbd, { rng, std: INIT_STD / Math.sqrt(2 * nLayer) });
  }

  forward(x: Tensor): Tensor {
    const [b, t, c] = x.shape;
    const flat = x.reshape([b * t, c]);
    const hidden = this.fc.forward(flat).gelu();
    return this.proj.forward(hidden).reshape([b, t, c]);
  }
}

/**
 * SwiGLU feed-forward (Shazeer, 2020): down(silu(gate(x)) * up(x)).
 *
 * The multiplicative gate lets each channel suppress or pass its partner, which
 * a single GELU projection cannot express. It buys a consistent loss improvement
 * over GELU at matched parameters — one of the few free lunches in this shape.
 */
class SwiGLU extends Module {
  private readonly gate: Linear;
  private readonly up: Linear;
  private readonly down: Linear;

  constructor(config: GPTConfig, rng: Rng) {
    super();
    const { nEmbd, nLayer } = config;
    const hidden = mlpHidden(config);
    this.gate = new Linear(nEmbd, hidden, { rng, std: INIT_STD, bias: false });
    this.up = new Linear(nEmbd, hidden, { rng, std: INIT_STD, bias: false });
    this.down = new Linear(hidden, nEmbd, { rng, std: INIT_STD / Math.sqrt(2 * nLayer), bias: false });
  }

  forward(x: Tensor): Tensor {
    const [b, t, c] = x.shape;
    const flat = x.reshape([b * t, c]);
    const hidden = this.gate.forward(flat).silu().mul(this.up.forward(flat));
    return this.down.forward(hidden).reshape([b, t, c]);
  }
}

/** Transformer block: attention and MLP, each with a pre-norm and a residual. */
class Block extends Module {
  private readonly ln1: Norm;
  private readonly attn: CausalSelfAttention;
  private readonly ln2: Norm;
  private readonly mlp: GeluMLP | SwiGLU;

  constructor(config: GPTConfig, rng: Rng) {
    super();
    this.ln1 = makeNorm(config, config.nEmbd);
    this.attn = new CausalSelfAttention(config, rng);
    this.ln2 = makeNorm(config, config.nEmbd);
    this.mlp = archOf(config) === "modern" ? new SwiGLU(config, rng) : new GeluMLP(config, rng);
  }

  forward(x: Tensor): Tensor {
    const h = x.add(this.attn.forward(this.ln1.forward(x)));
    return h.add(this.mlp.forward(this.ln2.forward(h)));
  }
}

export class GPT extends Module {
  readonly config: GPTConfig;
  /** Token embedding table [vocabSize, nEmbd]; also serves as the tied output head. */
  private readonly wte: Tensor;
  /** Learned positional embedding table [blockSize, nEmbd]. Null under RoPE. */
  private readonly wpe: Tensor | null;
  private readonly blocks: Block[];
  private readonly lnf: Norm;

  constructor(config: GPTConfig, rng: Rng = defaultRng) {
    super();
    this.config = config;
    this.wte = Tensor.randn([config.vocabSize, config.nEmbd], {
      requiresGrad: true,
      rng,
      std: INIT_STD,
    });
    this.wpe =
      archOf(config) === "modern"
        ? null
        : Tensor.randn([config.blockSize, config.nEmbd], { requiresGrad: true, rng, std: INIT_STD });
    this.blocks = Array.from({ length: config.nLayer }, () => new Block(config, rng));
    this.lnf = makeNorm(config, config.nEmbd);
  }

  /** Total trainable scalar count. */
  numParams(): number {
    return this.parameters().reduce((sum, p) => sum + p.size, 0);
  }

  /**
   * Runs the model over a batch of token ids.
   *
   * `idx` is a [batch, time] tensor whose values are integer token ids
   * (stored as floats). Returns logits shaped [batch * time, vocabSize],
   * flattened so the rows line up with `crossEntropyLogits` targets.
   */
  forward(idx: Tensor): Tensor {
    if (idx.ndim !== 2) throw new Error(`GPT.forward expects a [batch, time] tensor, got ${idx.ndim}-D`);
    const [b, t] = idx.shape;
    const { blockSize, nEmbd } = this.config;
    if (t > blockSize) throw new Error(`Sequence length ${t} exceeds block size ${blockSize}`);

    const tokenIds = new Int32Array(b * t);
    for (let i = 0; i < b * t; i++) tokenIds[i] = idx.data[i];

    const tokenEmb = this.wte.gatherRows(tokenIds);
    let x = tokenEmb;
    if (this.wpe) {
      // Positions repeat 0..t-1 for every sequence in the batch.
      const posIds = new Int32Array(b * t);
      for (let i = 0; i < b * t; i++) posIds[i] = i % t;
      x = x.add(this.wpe.gatherRows(posIds));
    }

    let h = x.reshape([b, t, nEmbd]);
    for (const block of this.blocks) h = block.forward(h);
    const out = this.lnf.forward(h).reshape([b * t, nEmbd]);

    // Output head shares weights with the token embedding (GPT-2 weight tying);
    // gradients from both uses accumulate into `wte`.
    return out.matmul(this.wte.transpose());
  }

  /** Cross-entropy of the model's predictions against the next-token targets. */
  loss(idx: Tensor, targets: ArrayLike<number>): Tensor {
    return this.forward(idx).crossEntropyLogits(targets);
  }

  /**
   * Autoregressively samples `maxNewTokens` continuations of `prompt`.
   * The context is cropped to the last `blockSize` tokens each step.
   * `onToken` is called with each sampled id; returning `false` stops
   * generation early (e.g. when a stop sequence appears in the decoded text).
   */
  generate(
    prompt: number[],
    maxNewTokens: number,
    opts: { temperature?: number; topK?: number; rng?: Rng; onToken?: (id: number) => boolean | void } = {},
  ): number[] {
    const { temperature = 1.0, topK, rng = defaultRng, onToken } = opts;
    const { blockSize, vocabSize } = this.config;
    const ids = [...prompt];

    for (let step = 0; step < maxNewTokens; step++) {
      const context = ids.slice(Math.max(0, ids.length - blockSize));
      const t = context.length;
      const idx = new Tensor(Float32Array.from(context), [1, t]);
      const logits = this.forward(idx); // [t, vocabSize]

      // Only the final position predicts the next token.
      const last = new Float32Array(vocabSize);
      const offset = (t - 1) * vocabSize;
      for (let j = 0; j < vocabSize; j++) last[j] = logits.data[offset + j] / temperature;

      if (topK && topK < vocabSize) {
        const threshold = [...last].sort((a, b) => b - a)[topK - 1];
        for (let j = 0; j < vocabSize; j++) if (last[j] < threshold) last[j] = -Infinity;
      }

      const next = sampleFromLogits(last, rng);
      ids.push(next);
      if (onToken && onToken(next) === false) break;
    }
    return ids;
  }
}

/** Draws one index from a softmax over `logits`. */
function sampleFromLogits(logits: Float32Array, rng: Rng): number {
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  let sum = 0;
  const probs = new Float32Array(logits.length);
  for (let j = 0; j < logits.length; j++) {
    const e = Math.exp(logits[j] - max);
    probs[j] = e;
    sum += e;
  }
  let r = rng() * sum;
  for (let j = 0; j < probs.length; j++) {
    r -= probs[j];
    if (r <= 0) return j;
  }
  return probs.length - 1;
}
