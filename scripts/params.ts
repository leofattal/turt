/**
 * Flat views over a model's parameters and gradients.
 *
 * Data-parallel training moves parameters out to workers and gradients back,
 * so both sides need one contiguous Float32Array with a stable layout. The
 * layout is simply `model.parameters()` order, which is deterministic for a
 * given config.
 */

import type { Tensor } from "../src/math/tensor.js";

export function totalSize(params: Tensor[]): number {
  return params.reduce((sum, p) => sum + p.size, 0);
}

/** Copies all parameter values into one contiguous array. */
export function flattenParams(params: Tensor[]): Float32Array {
  const flat = new Float32Array(totalSize(params));
  let offset = 0;
  for (const p of params) {
    flat.set(p.data, offset);
    offset += p.size;
  }
  return flat;
}

/** Writes a flat array back into the model's parameters. */
export function loadParams(params: Tensor[], flat: Float32Array): void {
  let offset = 0;
  for (const p of params) {
    p.data.set(flat.subarray(offset, offset + p.size));
    offset += p.size;
  }
}

/** Copies all gradients into one contiguous array (missing grads read as zero). */
export function flattenGrads(params: Tensor[]): Float32Array {
  const flat = new Float32Array(totalSize(params));
  let offset = 0;
  for (const p of params) {
    if (p.grad) flat.set(p.grad, offset);
    offset += p.size;
  }
  return flat;
}

/** Writes a flat gradient array into the model's `grad` buffers. */
export function loadGrads(params: Tensor[], flat: Float32Array): void {
  let offset = 0;
  for (const p of params) {
    p.grad ??= new Float32Array(p.size);
    p.grad.set(flat.subarray(offset, offset + p.size));
    offset += p.size;
  }
}
