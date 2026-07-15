/**
 * GPT — a decoder-only transformer in the GPT-2 / nanoGPT shape, built
 * entirely on Turt's tensor engine. Every layer composes primitive
 * differentiable ops, so autodiff supplies the whole backward pass.
 *
 * Architecture (pre-LayerNorm, as in GPT-2):
 *
 *   idx -> token embedding + learned positional embedding
 *       -> nLayer x [ x + attn(ln1(x)) ; x + mlp(ln2(x)) ]
 *       -> final LayerNorm
 *       -> lm_head (weights tied to the token embedding)
 *
 * Dropout is omitted: at the scale that is trainable on CPU the model is
 * heavily data-bound rather than overfitting, and skipping it buys compute.
 */

import { Tensor } from "../math/tensor.js";
import { type Rng, defaultRng } from "../math/random.js";
import { Module } from "../nn/module.js";
import { Linear } from "../nn/linear.js";
import { LayerNorm } from "../nn/layernorm.js";

export interface GPTConfig {
  vocabSize: number;
  blockSize: number;
  nLayer: number;
  nHead: number;
  nEmbd: number;
}

/** GPT-2 initializes most weights from N(0, 0.02). */
const INIT_STD = 0.02;

/** Multi-head causal self-attention over an [batch, time, channels] input. */
class CausalSelfAttention extends Module {
  private readonly q: Linear;
  private readonly k: Linear;
  private readonly v: Linear;
  private readonly proj: Linear;
  private readonly nHead: number;
  private readonly headDim: number;

  constructor(config: GPTConfig, rng: Rng) {
    super();
    const { nEmbd, nHead, nLayer } = config;
    if (nEmbd % nHead !== 0) {
      throw new Error(`nEmbd (${nEmbd}) must be divisible by nHead (${nHead})`);
    }
    this.nHead = nHead;
    this.headDim = nEmbd / nHead;
    this.q = new Linear(nEmbd, nEmbd, { rng, std: INIT_STD });
    this.k = new Linear(nEmbd, nEmbd, { rng, std: INIT_STD });
    this.v = new Linear(nEmbd, nEmbd, { rng, std: INIT_STD });
    // Residual projections are down-scaled so the residual stream variance
    // stays ~constant with depth (GPT-2's 1/sqrt(2 * nLayer) rule).
    this.proj = new Linear(nEmbd, nEmbd, { rng, std: INIT_STD / Math.sqrt(2 * nLayer) });
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

    const q = this.splitHeads(this.q.forward(flat), b, t);
    const k = this.splitHeads(this.k.forward(flat), b, t);
    const v = this.splitHeads(this.v.forward(flat), b, t);

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

/** Position-wise feed-forward network: c -> 4c -> GELU -> c. */
class MLP extends Module {
  private readonly fc: Linear;
  private readonly proj: Linear;

  constructor(config: GPTConfig, rng: Rng) {
    super();
    const { nEmbd, nLayer } = config;
    this.fc = new Linear(nEmbd, 4 * nEmbd, { rng, std: INIT_STD });
    this.proj = new Linear(4 * nEmbd, nEmbd, { rng, std: INIT_STD / Math.sqrt(2 * nLayer) });
  }

  forward(x: Tensor): Tensor {
    const [b, t, c] = x.shape;
    const flat = x.reshape([b * t, c]);
    const hidden = this.fc.forward(flat).gelu();
    return this.proj.forward(hidden).reshape([b, t, c]);
  }
}

/** Transformer block: attention and MLP, each with a pre-norm and a residual. */
class Block extends Module {
  private readonly ln1: LayerNorm;
  private readonly attn: CausalSelfAttention;
  private readonly ln2: LayerNorm;
  private readonly mlp: MLP;

  constructor(config: GPTConfig, rng: Rng) {
    super();
    this.ln1 = new LayerNorm(config.nEmbd);
    this.attn = new CausalSelfAttention(config, rng);
    this.ln2 = new LayerNorm(config.nEmbd);
    this.mlp = new MLP(config, rng);
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
  /** Learned positional embedding table [blockSize, nEmbd]. */
  private readonly wpe: Tensor;
  private readonly blocks: Block[];
  private readonly lnf: LayerNorm;

  constructor(config: GPTConfig, rng: Rng = defaultRng) {
    super();
    this.config = config;
    this.wte = Tensor.randn([config.vocabSize, config.nEmbd], {
      requiresGrad: true,
      rng,
      std: INIT_STD,
    });
    this.wpe = Tensor.randn([config.blockSize, config.nEmbd], {
      requiresGrad: true,
      rng,
      std: INIT_STD,
    });
    this.blocks = Array.from({ length: config.nLayer }, () => new Block(config, rng));
    this.lnf = new LayerNorm(config.nEmbd);
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
    // Positions repeat 0..t-1 for every sequence in the batch.
    const posIds = new Int32Array(b * t);
    for (let i = 0; i < b * t; i++) posIds[i] = i % t;

    const tokenEmb = this.wte.gatherRows(tokenIds);
    const posEmb = this.wpe.gatherRows(posIds);

    let x = tokenEmb.add(posEmb).reshape([b, t, nEmbd]);
    for (const block of this.blocks) x = block.forward(x);
    x = this.lnf.forward(x).reshape([b * t, nEmbd]);

    // Output head shares weights with the token embedding (GPT-2 weight tying);
    // gradients from both uses accumulate into `wte`.
    return x.matmul(this.wte.transpose());
  }

  /** Cross-entropy of the model's predictions against the next-token targets. */
  loss(idx: Tensor, targets: ArrayLike<number>): Tensor {
    return this.forward(idx).crossEntropyLogits(targets);
  }

  /**
   * Autoregressively samples `maxNewTokens` continuations of `prompt`.
   * The context is cropped to the last `blockSize` tokens each step.
   */
  generate(
    prompt: number[],
    maxNewTokens: number,
    opts: { temperature?: number; topK?: number; rng?: Rng } = {},
  ): number[] {
    const { temperature = 1.0, topK, rng = defaultRng } = opts;
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

      ids.push(sampleFromLogits(last, rng));
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
