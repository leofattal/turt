import { Tensor } from "../math/tensor.js";
import { Module } from "./module.js";

/**
 * Root-mean-square normalization (Zhang & Sennrich, 2019).
 *
 * LayerNorm centers a row and then rescales it. RMSNorm skips the centering and
 * rescales by the row's RMS alone, with a gain but no bias. It matches LayerNorm's
 * quality while dropping a reduction, a subtraction, and a whole parameter vector
 * per norm — which is why essentially every LM since Llama uses it.
 *
 * Built from primitive tensor ops, so autodiff supplies the backward pass.
 */
export class RMSNorm extends Module {
  readonly gamma: Tensor;
  readonly eps: number;

  constructor(normalizedDim: number, eps = 1e-5) {
    super();
    this.gamma = Tensor.ones([normalizedDim], true);
    this.eps = eps;
  }

  forward(input: Tensor): Tensor {
    const meanSquare = input.mul(input).mean(-1, true);
    return input.div(meanSquare.addScalar(this.eps).sqrt()).mul(this.gamma);
  }
}
