import { Tensor } from "../math/tensor.js";
import { Optimizer } from "./optimizer.js";

export interface AdamOptions {
  lr?: number;
  beta1?: number;
  beta2?: number;
  eps?: number;
  /** Decoupled weight decay (AdamW). 0 disables it, recovering plain Adam. */
  weightDecay?: number;
}

/** Adam with bias correction; set weightDecay > 0 for AdamW behavior. */
export class Adam extends Optimizer {
  readonly beta1: number;
  readonly beta2: number;
  readonly eps: number;
  readonly weightDecay: number;
  private t = 0;
  private m = new Map<Tensor, Float32Array>();
  private v = new Map<Tensor, Float32Array>();

  constructor(params: Tensor[], opts: AdamOptions = {}) {
    super(params, opts.lr ?? 0.001);
    this.beta1 = opts.beta1 ?? 0.9;
    this.beta2 = opts.beta2 ?? 0.999;
    this.eps = opts.eps ?? 1e-8;
    this.weightDecay = opts.weightDecay ?? 0;
  }

  step(): void {
    this.t++;
    const bc1 = 1 - Math.pow(this.beta1, this.t);
    const bc2 = 1 - Math.pow(this.beta2, this.t);
    for (const p of this.params) {
      const g = p.grad;
      if (!g) continue;
      let m = this.m.get(p);
      let v = this.v.get(p);
      if (!m || !v) {
        m = new Float32Array(p.size);
        v = new Float32Array(p.size);
        this.m.set(p, m);
        this.v.set(p, v);
      }
      for (let i = 0; i < p.size; i++) {
        m[i] = this.beta1 * m[i] + (1 - this.beta1) * g[i];
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * g[i] * g[i];
        const mHat = m[i] / bc1;
        const vHat = v[i] / bc2;
        p.data[i] -= this.lr * (mHat / (Math.sqrt(vHat) + this.eps) + this.weightDecay * p.data[i]);
      }
    }
  }
}

/** AdamW: Adam with decoupled weight decay on by default. */
export class AdamW extends Adam {
  constructor(params: Tensor[], opts: AdamOptions = {}) {
    super(params, { weightDecay: 0.01, ...opts });
  }
}
