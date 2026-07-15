import { Tensor } from "../math/tensor.js";

/**
 * Base class for all neural-network building blocks.
 *
 * `parameters()` discovers trainable tensors by walking the module's own
 * fields (tensors with requiresGrad, nested modules, and arrays of either),
 * so layers rarely need to override it.
 */
export abstract class Module {
  abstract forward(input: Tensor): Tensor;

  parameters(): Tensor[] {
    const params: Tensor[] = [];
    const seen = new Set<Tensor>();
    const collect = (value: unknown): void => {
      if (value instanceof Tensor) {
        if (value.requiresGrad && !seen.has(value)) {
          seen.add(value);
          params.push(value);
        }
      } else if (value instanceof Module) {
        for (const p of value.parameters()) {
          if (!seen.has(p)) {
            seen.add(p);
            params.push(p);
          }
        }
      } else if (Array.isArray(value)) {
        for (const v of value) collect(v);
      }
    };
    for (const value of Object.values(this)) collect(value);
    return params;
  }
}

/** Chains modules: output of each feeds the next. */
export class Sequential extends Module {
  readonly layers: Module[];

  constructor(...layers: Module[]) {
    super();
    this.layers = layers;
  }

  forward(input: Tensor): Tensor {
    let x = input;
    for (const layer of this.layers) x = layer.forward(x);
    return x;
  }
}
