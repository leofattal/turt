import { Tensor } from "../math/tensor.js";
import type { Module } from "../nn/module.js";
import { mseLoss } from "../nn/losses.js";
import type { Optimizer } from "../optim/optimizer.js";

export interface Batch {
  input: Tensor;
  target: Tensor;
}

export interface FitOptions {
  epochs: number;
  lossFn?: (prediction: Tensor, target: Tensor) => Tensor;
  /** Called after each epoch with the mean training loss. Return false to stop early. */
  onEpochEnd?: (epoch: number, meanLoss: number) => boolean | void;
}

/**
 * Minimal mini-batch training loop: forward, loss, backward, step.
 *
 * Extension points (see PRD "Training System"): dataset streaming (accept an
 * async iterable of batches), checkpointing, validation/early-stopping
 * callbacks, distributed hooks, and experiment tracking all belong here
 * rather than inside models or optimizers.
 */
export class Trainer {
  constructor(
    private readonly model: Module,
    private readonly optimizer: Optimizer,
  ) {}

  fit(batches: Batch[], opts: FitOptions): number {
    const lossFn = opts.lossFn ?? mseLoss;
    let meanLoss = Number.POSITIVE_INFINITY;
    for (let epoch = 0; epoch < opts.epochs; epoch++) {
      let total = 0;
      for (const batch of batches) {
        this.optimizer.zeroGrad();
        const prediction = this.model.forward(batch.input);
        const loss = lossFn(prediction, batch.target);
        loss.backward();
        this.optimizer.step();
        total += loss.item();
      }
      meanLoss = total / batches.length;
      if (opts.onEpochEnd?.(epoch, meanLoss) === false) break;
    }
    return meanLoss;
  }
}
