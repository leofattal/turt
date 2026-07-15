import { Tensor } from "../math/tensor.js";
import { type Rng, defaultRng } from "../math/random.js";
import { Module } from "./module.js";

/** Fully-connected layer: y = x @ W + b, with W [inFeatures, outFeatures]. */
export class Linear extends Module {
  readonly weight: Tensor;
  readonly bias: Tensor | null;

  constructor(
    inFeatures: number,
    outFeatures: number,
    opts: { bias?: boolean; rng?: Rng; std?: number } = {},
  ) {
    super();
    const { bias = true, rng = defaultRng } = opts;
    // Xavier/Glorot-style scaling keeps activations well-conditioned at init.
    // Callers that follow a different init scheme (e.g. GPT-2's fixed 0.02,
    // with residual projections scaled by 1/sqrt(2 * nLayer)) pass `std`.
    const std = opts.std ?? Math.sqrt(2 / (inFeatures + outFeatures));
    this.weight = Tensor.randn([inFeatures, outFeatures], { requiresGrad: true, rng, std });
    this.bias = bias ? Tensor.zeros([outFeatures], true) : null;
  }

  forward(input: Tensor): Tensor {
    const out = input.matmul(this.weight);
    return this.bias ? out.add(this.bias) : out;
  }
}
