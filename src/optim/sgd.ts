import { Tensor } from "../math/tensor.js";
import { Optimizer } from "./optimizer.js";

/** Stochastic gradient descent, with optional classical momentum. */
export class SGD extends Optimizer {
  readonly momentum: number;
  private velocities = new Map<Tensor, Float32Array>();

  constructor(params: Tensor[], opts: { lr?: number; momentum?: number } = {}) {
    super(params, opts.lr ?? 0.01);
    this.momentum = opts.momentum ?? 0;
  }

  step(): void {
    for (const p of this.params) {
      const g = p.grad;
      if (!g) continue;
      if (this.momentum > 0) {
        let v = this.velocities.get(p);
        if (!v) {
          v = new Float32Array(p.size);
          this.velocities.set(p, v);
        }
        for (let i = 0; i < p.size; i++) {
          v[i] = this.momentum * v[i] + g[i];
          p.data[i] -= this.lr * v[i];
        }
      } else {
        for (let i = 0; i < p.size; i++) p.data[i] -= this.lr * g[i];
      }
    }
  }
}
