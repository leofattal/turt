export { Optimizer } from "./optimizer.js";
export { SGD } from "./sgd.js";
export { Adam, AdamW, type AdamOptions } from "./adam.js";
export { RMSProp } from "./rmsprop.js";
export { LRScheduler, StepLR, CosineAnnealingLR } from "./schedulers.js";
export { clipGradNorm } from "./clip.js";

// Planned (see PRD "Optimizers"): mixed-precision training support once a
// lower-precision buffer type lands in the math engine.
