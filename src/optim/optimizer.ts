import { Tensor } from "../math/tensor.js";

/** Base class for gradient-based optimizers. */
export abstract class Optimizer {
  lr: number;
  readonly params: Tensor[];

  constructor(params: Tensor[], lr: number) {
    this.params = params;
    this.lr = lr;
  }

  zeroGrad(): void {
    for (const p of this.params) p.zeroGrad();
  }

  /** Applies one update using the gradients currently stored on the params. */
  abstract step(): void;
}
