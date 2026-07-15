/**
 * GPU-resident GPT — the same architecture as src/models/gpt.ts, but built on
 * `GpuTensor` so the entire forward and backward pass stays in GPU memory. Only
 * the token ids go in and the scalar loss comes out each step.
 *
 * The forward mirrors the CPU model op-for-op (pre-LayerNorm blocks, causal
 * multi-head attention, GELU MLP, weight-tied head), which is what lets the two
 * be validated against each other: load identical weights, and every gradient
 * matches. Activations are kept flattened to [N, nEmbd] with N = batch*time,
 * expanding to 4-D only inside attention where the head split needs it.
 */

import { GpuEngine, GpuTensor } from "../infer/webgpu/autograd.js";
import type { GPTConfig } from "./gpt.js";

const INIT_STD = 0.02;

interface Block {
  ln1g: GpuTensor; ln1b: GpuTensor;
  qW: GpuTensor; qb: GpuTensor;
  kW: GpuTensor; kb: GpuTensor;
  vW: GpuTensor; vb: GpuTensor;
  projW: GpuTensor; projb: GpuTensor;
  ln2g: GpuTensor; ln2b: GpuTensor;
  fcW: GpuTensor; fcb: GpuTensor;
  mpW: GpuTensor; mpb: GpuTensor;
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
  readonly wpe: GpuTensor;
  readonly lnfg: GpuTensor;
  readonly lnfb: GpuTensor;
  private readonly headScale: number;

  constructor(readonly engine: GpuEngine, readonly config: GPTConfig) {
    const { vocabSize, blockSize, nEmbd, nLayer, nHead } = config;
    if (nEmbd % nHead !== 0) throw new Error("nEmbd must be divisible by nHead");
    this.headScale = 1 / Math.sqrt(nEmbd / nHead);
    let seed = 1;
    const param = (n: number, shape: number[], std: number): GpuTensor =>
      engine.tensor(gaussianArray(n, seed++, std), shape, true);
    const zeros = (n: number, shape: number[]): GpuTensor => engine.tensor(new Float32Array(n), shape, true);
    const ones = (n: number, shape: number[]): GpuTensor => engine.tensor(new Float32Array(n).fill(1), shape, true);

    this.wte = param(vocabSize * nEmbd, [vocabSize, nEmbd], INIT_STD);
    this.wpe = param(blockSize * nEmbd, [blockSize, nEmbd], INIT_STD);
    const resStd = INIT_STD / Math.sqrt(2 * nLayer);
    for (let i = 0; i < nLayer; i++) {
      this.blocks.push({
        ln1g: ones(nEmbd, [nEmbd]), ln1b: zeros(nEmbd, [nEmbd]),
        qW: param(nEmbd * nEmbd, [nEmbd, nEmbd], INIT_STD), qb: zeros(nEmbd, [nEmbd]),
        kW: param(nEmbd * nEmbd, [nEmbd, nEmbd], INIT_STD), kb: zeros(nEmbd, [nEmbd]),
        vW: param(nEmbd * nEmbd, [nEmbd, nEmbd], INIT_STD), vb: zeros(nEmbd, [nEmbd]),
        projW: param(nEmbd * nEmbd, [nEmbd, nEmbd], resStd), projb: zeros(nEmbd, [nEmbd]),
        ln2g: ones(nEmbd, [nEmbd]), ln2b: zeros(nEmbd, [nEmbd]),
        fcW: param(nEmbd * 4 * nEmbd, [nEmbd, 4 * nEmbd], INIT_STD), fcb: zeros(4 * nEmbd, [4 * nEmbd]),
        mpW: param(4 * nEmbd * nEmbd, [4 * nEmbd, nEmbd], resStd), mpb: zeros(nEmbd, [nEmbd]),
      });
    }
    this.lnfg = ones(nEmbd, [nEmbd]);
    this.lnfb = zeros(nEmbd, [nEmbd]);
  }

  /** All trainable tensors, in the same order as the CPU model's parameters(). */
  parameters(): GpuTensor[] {
    const ps: GpuTensor[] = [this.wte, this.wpe];
    for (const b of this.blocks) {
      ps.push(b.ln1g, b.ln1b, b.qW, b.qb, b.kW, b.kb, b.vW, b.vb, b.projW, b.projb,
        b.ln2g, b.ln2b, b.fcW, b.fcb, b.mpW, b.mpb);
    }
    ps.push(this.lnfg, this.lnfb);
    return ps;
  }

  numParams(): number {
    return this.parameters().reduce((s, p) => s + p.size, 0);
  }

  private attention(xln: GpuTensor, b: number, t: number): GpuTensor {
    const { nEmbd, nHead } = this.config;
    const hd = nEmbd / nHead;
    const split = (p: GpuTensor): GpuTensor =>
      p.reshape([b, t, nHead, hd]).permute([0, 2, 1, 3]).reshape([b * nHead, t, hd]);
    const blk = this.currentBlock;
    const q = split(xln.matmul(blk.qW).add(blk.qb));
    const k = split(xln.matmul(blk.kW).add(blk.kb));
    const v = split(xln.matmul(blk.vW).add(blk.vb));
    const scores = q.bmmBT(k).scale(this.headScale).maskedFillCausal().softmax();
    const att = scores
      .bmm(v)
      .reshape([b, nHead, t, hd])
      .permute([0, 2, 1, 3])
      .reshape([b * t, nEmbd]);
    return att.matmul(blk.projW).add(blk.projb);
  }

  private currentBlock!: Block;

  /**
   * Forward pass returning the scalar cross-entropy loss.
   * `tokenIds` are the [batch*time] input ids; `targets` the next-token ids.
   */
  loss(tokenIds: Uint32Array, targets: Uint32Array, b: number, t: number): GpuTensor {
    const posIds = new Uint32Array(b * t);
    for (let i = 0; i < b * t; i++) posIds[i] = i % t;

    let x = this.wte.gatherRows(tokenIds).add(this.wpe.gatherRows(posIds)); // [N, E]
    for (const blk of this.blocks) {
      this.currentBlock = blk;
      const h = x.add(this.attention(x.layerNorm(blk.ln1g, blk.ln1b), b, t));
      const hln = h.layerNorm(blk.ln2g, blk.ln2b);
      const mlp = hln.matmul(blk.fcW).add(blk.fcb).gelu().matmul(blk.mpW).add(blk.mpb);
      x = h.add(mlp);
    }
    const xf = x.layerNorm(this.lnfg, this.lnfb);
    const logits = xf.matmulBT(this.wte); // [N, V], weight-tied head
    return logits.crossEntropyLogits(targets);
  }
}
