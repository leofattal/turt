/**
 * GPU-resident GPT — the same architecture as src/models/gpt.ts, but built on
 * `GpuTensor` so the entire forward and backward pass stays in GPU memory. Only
 * the token ids go in and the scalar loss comes out each step.
 *
 * The forward mirrors the CPU model op-for-op — for both architectures ("gpt2":
 * LayerNorm/GELU/learned positions/biases; "modern": RMSNorm/SwiGLU/RoPE/no
 * biases) — which is what lets the two be validated against each other: load
 * identical weights, and every gradient matches. `parameters()` returns tensors
 * in exactly the CPU model's field-walk order so checkpoints are interchangeable.
 *
 * Activations are kept flattened to [N, nEmbd] with N = batch*time, expanding to
 * 4-D only inside attention where the head split needs it.
 */

import { GpuEngine, GpuTensor } from "../infer/webgpu/autograd.js";
import { type GPTConfig, archOf, mlpHidden } from "./gpt.js";

const INIT_STD = 0.02;

/** Biases and `wpe` are absent under "modern"; `fcW/mpW` vs `gateW/upW/downW` swap with the MLP. */
interface Block {
  ln1g: GpuTensor; ln1b?: GpuTensor;
  qW: GpuTensor; qb?: GpuTensor;
  kW: GpuTensor; kb?: GpuTensor;
  vW: GpuTensor; vb?: GpuTensor;
  projW: GpuTensor; projb?: GpuTensor;
  ln2g: GpuTensor; ln2b?: GpuTensor;
  fcW?: GpuTensor; fcb?: GpuTensor;
  mpW?: GpuTensor; mpb?: GpuTensor;
  gateW?: GpuTensor; upW?: GpuTensor; downW?: GpuTensor;
}

function gaussianArray(n: number, seed: number, std: number): Float32Array {
  // Simple seeded normal (mulberry32 + Box-Muller); weights are usually
  // overwritten from a checkpoint or the CPU model, so exact stream is moot.
  let s = seed >>> 0;
  const rng = (): number => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let u = 0;
    while (u === 0) u = rng();
    out[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng()) * std;
  }
  return out;
}

export class GpuGPT {
  readonly blocks: Block[] = [];
  readonly wte: GpuTensor;
  readonly wpe: GpuTensor | null;
  readonly lnfg: GpuTensor;
  readonly lnfb?: GpuTensor;
  private readonly headScale: number;
  private readonly modern: boolean;

  constructor(readonly engine: GpuEngine, readonly config: GPTConfig, opts: { skipInit?: boolean } = {}) {
    const { vocabSize, blockSize, nEmbd, nLayer, nHead } = config;
    if (nEmbd % nHead !== 0) throw new Error("nEmbd must be divisible by nHead");
    this.modern = archOf(config) === "modern";
    this.headScale = 1 / Math.sqrt(nEmbd / nHead);
    const hidden = mlpHidden(config);
    let seed = 1;
    // skipInit: allocate zeroed buffers without uploading anything — for
    // callers that overwrite every parameter right away (trainer seeds from
    // the CPU model). Staging ~134MB of init data in one synchronous burst
    // overflows the host-visible heap on discrete GPUs before any poll can
    // reclaim it, which poisons the device for the whole run.
    const raw = (n: number, shape: number[]): GpuTensor => new GpuTensor(engine, engine.buffer(n), shape, true);
    const param = (n: number, shape: number[], std: number): GpuTensor =>
      opts.skipInit ? raw(n, shape) : engine.tensor(gaussianArray(n, seed++, std), shape, true);
    const zeros = (n: number, shape: number[]): GpuTensor =>
      opts.skipInit ? raw(n, shape) : engine.tensor(new Float32Array(n), shape, true);
    const ones = (n: number, shape: number[]): GpuTensor =>
      opts.skipInit ? raw(n, shape) : engine.tensor(new Float32Array(n).fill(1), shape, true);

    this.wte = param(vocabSize * nEmbd, [vocabSize, nEmbd], INIT_STD);
    this.wpe = this.modern ? null : param(blockSize * nEmbd, [blockSize, nEmbd], INIT_STD);
    const resStd = INIT_STD / Math.sqrt(2 * nLayer);
    for (let i = 0; i < nLayer; i++) {
      const blk: Block = {
        ln1g: ones(nEmbd, [nEmbd]),
        qW: param(nEmbd * nEmbd, [nEmbd, nEmbd], INIT_STD),
        kW: param(nEmbd * nEmbd, [nEmbd, nEmbd], INIT_STD),
        vW: param(nEmbd * nEmbd, [nEmbd, nEmbd], INIT_STD),
        projW: param(nEmbd * nEmbd, [nEmbd, nEmbd], resStd),
        ln2g: ones(nEmbd, [nEmbd]),
      };
      if (this.modern) {
        blk.gateW = param(nEmbd * hidden, [nEmbd, hidden], INIT_STD);
        blk.upW = param(nEmbd * hidden, [nEmbd, hidden], INIT_STD);
        blk.downW = param(hidden * nEmbd, [hidden, nEmbd], resStd);
      } else {
        blk.ln1b = zeros(nEmbd, [nEmbd]);
        blk.qb = zeros(nEmbd, [nEmbd]);
        blk.kb = zeros(nEmbd, [nEmbd]);
        blk.vb = zeros(nEmbd, [nEmbd]);
        blk.projb = zeros(nEmbd, [nEmbd]);
        blk.ln2b = zeros(nEmbd, [nEmbd]);
        blk.fcW = param(nEmbd * hidden, [nEmbd, hidden], INIT_STD);
        blk.fcb = zeros(hidden, [hidden]);
        blk.mpW = param(hidden * nEmbd, [hidden, nEmbd], resStd);
        blk.mpb = zeros(nEmbd, [nEmbd]);
      }
      this.blocks.push(blk);
    }
    this.lnfg = ones(nEmbd, [nEmbd]);
    if (!this.modern) this.lnfb = zeros(nEmbd, [nEmbd]);
  }

  /** All trainable tensors, in the same order as the CPU model's parameters(). */
  parameters(): GpuTensor[] {
    const ps: GpuTensor[] = [this.wte];
    if (this.wpe) ps.push(this.wpe);
    for (const b of this.blocks) {
      if (this.modern) {
        ps.push(b.ln1g, b.qW, b.kW, b.vW, b.projW, b.ln2g, b.gateW!, b.upW!, b.downW!);
      } else {
        ps.push(b.ln1g, b.ln1b!, b.qW, b.qb!, b.kW, b.kb!, b.vW, b.vb!, b.projW, b.projb!,
          b.ln2g, b.ln2b!, b.fcW!, b.fcb!, b.mpW!, b.mpb!);
      }
    }
    ps.push(this.lnfg);
    if (this.lnfb) ps.push(this.lnfb);
    return ps;
  }

  numParams(): number {
    return this.parameters().reduce((s, p) => s + p.size, 0);
  }

  /** Pre-norm: RMSNorm under "modern", LayerNorm under "gpt2". */
  private norm(x: GpuTensor, gamma: GpuTensor, beta?: GpuTensor): GpuTensor {
    return this.modern ? x.rmsNorm(gamma) : x.layerNorm(gamma, beta!);
  }

  private attention(xln: GpuTensor, b: number, t: number): GpuTensor {
    const { nEmbd, nHead } = this.config;
    const hd = nEmbd / nHead;
    const split = (p: GpuTensor): GpuTensor =>
      p.reshape([b, t, nHead, hd]).permute([0, 2, 1, 3]).reshape([b * nHead, t, hd]);
    const blk = this.currentBlock;
    const lin = (w: GpuTensor, bias?: GpuTensor): GpuTensor => {
      const y = xln.matmul(w);
      return bias ? y.add(bias) : y;
    };
    let q = split(lin(blk.qW, blk.qb));
    let k = split(lin(blk.kW, blk.kb));
    const v = split(lin(blk.vW, blk.vb));
    if (this.modern) {
      q = q.rope();
      k = k.rope();
    }
    const scores = q.bmmBT(k).scale(this.headScale).maskedFillCausal().softmax();
    const att = scores
      .bmm(v)
      .reshape([b, nHead, t, hd])
      .permute([0, 2, 1, 3])
      .reshape([b * t, nEmbd]);
    const projected = att.matmul(blk.projW);
    return blk.projb ? projected.add(blk.projb) : projected;
  }

  private mlp(hln: GpuTensor): GpuTensor {
    const blk = this.currentBlock;
    if (this.modern) {
      const hidden = hln.matmul(blk.gateW!).silu().mul(hln.matmul(blk.upW!));
      return hidden.matmul(blk.downW!);
    }
    return hln.matmul(blk.fcW!).add(blk.fcb!).gelu().matmul(blk.mpW!).add(blk.mpb!);
  }

  private currentBlock!: Block;

  /**
   * Forward pass returning the scalar cross-entropy loss.
   * `tokenIds` are the [batch*time] input ids; `targets` the next-token ids.
   */
  loss(tokenIds: Uint32Array, targets: Uint32Array, b: number, t: number): GpuTensor {
    let x = this.wte.gatherRows(tokenIds); // [N, E]
    if (this.wpe) {
      const posIds = new Uint32Array(b * t);
      for (let i = 0; i < b * t; i++) posIds[i] = i % t;
      x = x.add(this.wpe.gatherRows(posIds));
    }
    for (const blk of this.blocks) {
      this.currentBlock = blk;
      const h = x.add(this.attention(this.norm(x, blk.ln1g, blk.ln1b), b, t));
      this.currentBlock = blk;
      x = h.add(this.mlp(this.norm(h, blk.ln2g, blk.ln2b)));
    }
    const xf = this.norm(x, this.lnfg, this.lnfb);
    const logits = xf.matmulBT(this.wte); // [N, V], weight-tied head
    return logits.crossEntropyLogits(targets);
  }
}
