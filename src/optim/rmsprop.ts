import { Tensor } from "../math/tensor.js";
import { Optimizer } from "./optimizer.js";

export class RMSProp extends Optimizer {
  readonly alpha: number;
  readonly eps: number;
  private sq = new Map<Tensor, Float32Array>();

  constructor(params: Tensor[], opts: { lr?: number; alpha?: number; eps?: number } = {}) {
    super(params, opts.lr ?? 0.01);
    this.alpha = opts.alpha ?? 0.99;
    this.eps = opts.eps ?? 1e-8;
  }

  step(): void {
    for (const p of this.params) {
      const g = p.grad;
      if (!g) continue;
      let s = this.sq.get(p);
      if (!s) {
        s = new Float32Array(p.size);
        this.sq.set(p, s);
      }
      for (let i = 0; i < p.size; i++) {
        s[i] = this.alpha * s[i] + (1 - this.alpha) * g[i] * g[i];
        p.data[i] -= (this.lr * g[i]) / (Math.sqrt(s[i]) + this.eps);
      }
    }
  }
}
