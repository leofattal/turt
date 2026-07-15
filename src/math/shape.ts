/**
 * Shape utilities for the math engine.
 *
 * Tensors are dense, contiguous, row-major. Broadcasting follows NumPy rules:
 * shapes are aligned right-to-left, and dimensions of size 1 stretch.
 */

export type Shape = readonly number[];

/** Total number of elements for a shape. The empty shape [] is a scalar (size 1). */
export function sizeOf(shape: Shape): number {
  let n = 1;
  for (const d of shape) n *= d;
  return n;
}

/** Row-major strides for a shape. */
export function stridesFor(shape: Shape): number[] {
  const strides = new Array<number>(shape.length).fill(1);
  for (let i = shape.length - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

export function shapesEqual(a: Shape, b: Shape): boolean {
  return a.length === b.length && a.every((d, i) => d === b[i]);
}

/** Resulting shape of broadcasting `a` with `b`. Throws when incompatible. */
export function broadcastShapes(a: Shape, b: Shape): number[] {
  const len = Math.max(a.length, b.length);
  const out = new Array<number>(len);
  for (let i = 0; i < len; i++) {
    const ad = a[a.length - 1 - i] ?? 1;
    const bd = b[b.length - 1 - i] ?? 1;
    if (ad !== bd && ad !== 1 && bd !== 1) {
      throw new Error(`Cannot broadcast shapes [${a.join(",")}] and [${b.join(",")}]`);
    }
    out[len - 1 - i] = Math.max(ad, bd);
  }
  return out;
}

/**
 * Returns a function mapping a flat index in `outShape` to the flat index of the
 * corresponding element in `shape` (right-aligned, size-1 dims broadcast).
 */
export function broadcastIndexer(outShape: Shape, shape: Shape): (outIndex: number) => number {
  if (shapesEqual(outShape, shape)) return (i) => i;
  const outStrides = stridesFor(outShape);
  const strides = stridesFor(shape);
  const offset = outShape.length - shape.length;
  return (outIndex: number) => {
    let idx = 0;
    let rem = outIndex;
    for (let d = 0; d < outShape.length; d++) {
      const coord = Math.floor(rem / outStrides[d]);
      rem -= coord * outStrides[d];
      const sd = d - offset;
      if (sd >= 0 && shape[sd] !== 1) idx += coord * strides[sd];
    }
    return idx;
  };
}

/** Normalizes a possibly-negative axis index against `ndim`. */
export function normalizeAxis(axis: number, ndim: number): number {
  const ax = axis < 0 ? ndim + axis : axis;
  if (ax < 0 || ax >= ndim) throw new Error(`Axis ${axis} out of range for ndim ${ndim}`);
  return ax;
}
