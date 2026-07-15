import type { Optimizer } from "./optimizer.js";

/**
 * Learning-rate schedulers mutate `optimizer.lr` in place. Call `step()`
 * once per epoch (or per iteration, depending on the schedule you want).
 */
export abstract class LRScheduler {
  protected readonly baseLr: number;
  protected stepCount = 0;

  constructor(protected readonly optimizer: Optimizer) {
    this.baseLr = optimizer.lr;
  }

  step(): void {
    this.stepCount++;
    this.optimizer.lr = this.computeLr();
  }

  protected abstract computeLr(): number;
}

/** Multiplies the learning rate by `gamma` every `stepSize` steps. */
export class StepLR extends LRScheduler {
  constructor(
    optimizer: Optimizer,
    private readonly stepSize: number,
    private readonly gamma = 0.1,
  ) {
    super(optimizer);
  }

  protected computeLr(): number {
    return this.baseLr * Math.pow(this.gamma, Math.floor(this.stepCount / this.stepSize));
  }
}

/** Cosine annealing from the base LR down to `etaMin` over `tMax` steps. */
export class CosineAnnealingLR extends LRScheduler {
  constructor(
    optimizer: Optimizer,
    private readonly tMax: number,
    private readonly etaMin = 0,
  ) {
    super(optimizer);
  }

  protected computeLr(): number {
    const t = Math.min(this.stepCount, this.tMax);
    return this.etaMin + ((this.baseLr - this.etaMin) * (1 + Math.cos((Math.PI * t) / this.tMax))) / 2;
  }
}
