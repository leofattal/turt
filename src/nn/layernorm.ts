import { Tensor } from "../math/tensor.js";
import { Module } from "./module.js";

/**
 * Layer normalization over the last dimension, with learnable gain/bias.
 * Built from primitive tensor ops, so gradients flow automatically.
 */
export class LayerNorm extends Module {
  readonly gamma: Tensor;
  readonly beta: Tensor;
  readonly eps: number;

  constructor(normalizedDim: number, eps = 1e-5) {
    super();
    this.gamma = Tensor.ones([normalizedDim], true);
    this.beta = Tensor.zeros([normalizedDim], true);
    this.eps = eps;
  }

  forward(input: Tensor): Tensor {
    const mu = input.mean(-1, true);
    const centered = input.sub(mu);
    const variance = centered.mul(centered).mean(-1, true);
    const norm = centered.div(variance.addScalar(this.eps).sqrt());
    return norm.mul(this.gamma).add(this.beta);
  }
}
