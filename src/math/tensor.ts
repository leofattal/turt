/**
 * Tensor: dense row-major Float32 tensor with reverse-mode automatic differentiation.
 *
 * Every differentiable op records a `GradCtx` (its parents plus a backward
 * closure) on the output tensor, forming the computational graph. Calling
 * `backward()` on a scalar output topologically sorts the graph and
 * accumulates gradients into each tensor's `grad` buffer.
 *
 * Design notes:
 * - CPU-only reference implementation; backends (WebGPU/CUDA) plug in later
 *   via the inference engine's backend interface (see src/infer).
 * - Gradient checkpointing and SIMD/WASM kernels are planned extension points;
 *   ops are funneled through a few helpers (unary/binary/reduce) so kernels
 *   can be swapped without touching the graph machinery.
 */

import {
  type Shape,
  sizeOf,
  stridesFor,
  shapesEqual,
  broadcastShapes,
  broadcastIndexer,
  normalizeAxis,
} from "./shape.js";
import { type Rng, defaultRng, gaussian } from "./random.js";

interface GradCtx {
  parents: Tensor[];
  backward: (outGrad: Float32Array) => void;
}

export class Tensor {
  readonly data: Float32Array;
  readonly shape: number[];
  requiresGrad: boolean;
  grad: Float32Array | null = null;
  private ctx: GradCtx | null = null;

  constructor(data: Float32Array | number[], shape: Shape, requiresGrad = false) {
    this.data = data instanceof Float32Array ? data : Float32Array.from(data);
    this.shape = [...shape];
    if (this.data.length !== sizeOf(shape)) {
      throw new Error(
        `Data length ${this.data.length} does not match shape [${shape.join(",")}] (size ${sizeOf(shape)})`,
      );
    }
    this.requiresGrad = requiresGrad;
  }

  get size(): number {
    return this.data.length;
  }

  get ndim(): number {
    return this.shape.length;
  }

  // ---------------------------------------------------------------- factories

  static zeros(shape: Shape, requiresGrad = false): Tensor {
    return new Tensor(new Float32Array(sizeOf(shape)), shape, requiresGrad);
  }

  static ones(shape: Shape, requiresGrad = false): Tensor {
    return Tensor.full(shape, 1, requiresGrad);
  }

  static full(shape: Shape, value: number, requiresGrad = false): Tensor {
    const data = new Float32Array(sizeOf(shape)).fill(value);
    return new Tensor(data, shape, requiresGrad);
  }

  static scalar(value: number, requiresGrad = false): Tensor {
    return new Tensor([value], [], requiresGrad);
  }

  static fromArray(values: number[], shape?: Shape, requiresGrad = false): Tensor {
    return new Tensor(values, shape ?? [values.length], requiresGrad);
  }

  /** Standard-normal init; pass a seeded rng for reproducibility. */
  static randn(shape: Shape, opts: { requiresGrad?: boolean; rng?: Rng; std?: number } = {}): Tensor {
    const { requiresGrad = false, rng = defaultRng, std = 1 } = opts;
    const data = new Float32Array(sizeOf(shape));
    for (let i = 0; i < data.length; i++) data[i] = gaussian(rng) * std;
    return new Tensor(data, shape, requiresGrad);
  }

  // ------------------------------------------------------------------ access

  /** Value of a single-element tensor. */
  item(): number {
    if (this.size !== 1) throw new Error(`item() requires a single-element tensor, got size ${this.size}`);
    return this.data[0];
  }

  toArray(): number[] {
    return Array.from(this.data);
  }

  // ------------------------------------------------------------ graph plumbing

  private static make(
    data: Float32Array,
    shape: Shape,
    parents: Tensor[],
    backward: (outGrad: Float32Array) => void,
  ): Tensor {
    const out = new Tensor(data, shape);
    if (parents.some((p) => p.requiresGrad)) {
      out.requiresGrad = true;
      out.ctx = { parents, backward };
    }
    return out;
  }

  private ensureGrad(): Float32Array {
    this.grad ??= new Float32Array(this.size);
    return this.grad;
  }

  /**
   * Reverse-mode backprop from this tensor. Without an explicit seed gradient,
   * the tensor must be a scalar (the usual loss case).
   */
  backward(seed?: Float32Array): void {
    if (!seed) {
      if (this.size !== 1) throw new Error("backward() without a seed gradient requires a scalar output");
      seed = new Float32Array([1]);
    }
    const g = this.ensureGrad();
    for (let i = 0; i < g.length; i++) g[i] += seed[i];

    // Iterative topological sort (post-order over the graph). Nodes are marked
    // visited at expansion time, not push time — marking on push reorders
    // diamond-shaped graphs and would consume a gradient before every child
    // has contributed to it.
    const topo: Tensor[] = [];
    const visited = new Set<Tensor>();
    const stack: Array<{ t: Tensor; expanded: boolean }> = [{ t: this, expanded: false }];
    while (stack.length > 0) {
      const frame = stack[stack.length - 1];
      if (frame.expanded) {
        stack.pop();
        topo.push(frame.t);
        continue;
      }
      if (visited.has(frame.t)) {
        stack.pop();
        continue;
      }
      visited.add(frame.t);
      frame.expanded = true;
      if (frame.t.ctx) {
        for (const p of frame.t.ctx.parents) {
          if (p.ctx && !visited.has(p)) stack.push({ t: p, expanded: false });
        }
      }
    }

    for (let i = topo.length - 1; i >= 0; i--) {
      const t = topo[i];
      if (t.ctx && t.grad) t.ctx.backward(t.grad);
    }
  }

  /** Clears the gradient buffer (typically via Optimizer.zeroGrad). */
  zeroGrad(): void {
    this.grad = null;
  }

  /** A copy that is disconnected from the graph. */
  detach(): Tensor {
    return new Tensor(this.data.slice(), this.shape);
  }

  // ------------------------------------------------------------- op helpers

  private static binary(
    a: Tensor,
    b: Tensor,
    f: (x: number, y: number) => number,
    dfdx: (x: number, y: number, g: number) => number,
    dfdy: (x: number, y: number, g: number) => number,
  ): Tensor {
    const outShape = broadcastShapes(a.shape, b.shape);
    const n = sizeOf(outShape);
    const ai = broadcastIndexer(outShape, a.shape);
    const bi = broadcastIndexer(outShape, b.shape);
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = f(a.data[ai(i)], b.data[bi(i)]);
    return Tensor.make(data, outShape, [a, b], (g) => {
      if (a.requiresGrad) {
        const ga = a.ensureGrad();
        for (let i = 0; i < n; i++) {
          const ia = ai(i);
          ga[ia] += dfdx(a.data[ia], b.data[bi(i)], g[i]);
        }
      }
      if (b.requiresGrad) {
        const gb = b.ensureGrad();
        for (let i = 0; i < n; i++) {
          const ib = bi(i);
          gb[ib] += dfdy(a.data[ai(i)], b.data[ib], g[i]);
        }
      }
    });
  }

  private static unary(
    a: Tensor,
    f: (x: number) => number,
    dfdx: (x: number, y: number, g: number) => number,
  ): Tensor {
    const n = a.size;
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = f(a.data[i]);
    return Tensor.make(data, a.shape, [a], (g) => {
      if (!a.requiresGrad) return;
      const ga = a.ensureGrad();
      for (let i = 0; i < n; i++) ga[i] += dfdx(a.data[i], data[i], g[i]);
    });
  }

  // ------------------------------------------------------------ element-wise

  add(b: Tensor): Tensor {
    return Tensor.binary(this, b, (x, y) => x + y, (_x, _y, g) => g, (_x, _y, g) => g);
  }

  sub(b: Tensor): Tensor {
    return Tensor.binary(this, b, (x, y) => x - y, (_x, _y, g) => g, (_x, _y, g) => -g);
  }

  mul(b: Tensor): Tensor {
    return Tensor.binary(this, b, (x, y) => x * y, (_x, y, g) => g * y, (x, _y, g) => g * x);
  }

  div(b: Tensor): Tensor {
    return Tensor.binary(
      this,
      b,
      (x, y) => x / y,
      (_x, y, g) => g / y,
      (x, y, g) => (-g * x) / (y * y),
    );
  }

  /** Convenience for op with a plain number (broadcast scalar). */
  addScalar(v: number): Tensor {
    return this.add(Tensor.scalar(v));
  }

  mulScalar(v: number): Tensor {
    return this.mul(Tensor.scalar(v));
  }

  neg(): Tensor {
    return Tensor.unary(this, (x) => -x, (_x, _y, g) => -g);
  }

  relu(): Tensor {
    return Tensor.unary(this, (x) => (x > 0 ? x : 0), (x, _y, g) => (x > 0 ? g : 0));
  }

  tanh(): Tensor {
    return Tensor.unary(this, Math.tanh, (_x, y, g) => g * (1 - y * y));
  }

  sigmoid(): Tensor {
    return Tensor.unary(this, (x) => 1 / (1 + Math.exp(-x)), (_x, y, g) => g * y * (1 - y));
  }

  exp(): Tensor {
    return Tensor.unary(this, Math.exp, (_x, y, g) => g * y);
  }

  log(): Tensor {
    return Tensor.unary(this, Math.log, (x, _y, g) => g / x);
  }

  sqrt(): Tensor {
    return Tensor.unary(this, Math.sqrt, (_x, y, g) => g / (2 * y));
  }

  pow(exponent: number): Tensor {
    return Tensor.unary(
      this,
      (x) => Math.pow(x, exponent),
      (x, _y, g) => g * exponent * Math.pow(x, exponent - 1),
    );
  }

  // -------------------------------------------------------------- reductions

  /**
   * Sum over all elements (axis = null) or along a single axis.
   * Negative axes count from the end.
   */
  sum(axis: number | null = null, keepDims = false): Tensor {
    if (axis === null) {
      let total = 0;
      for (let i = 0; i < this.size; i++) total += this.data[i];
      const self = this;
      return Tensor.make(new Float32Array([total]), [], [this], (g) => {
        if (!self.requiresGrad) return;
        const ga = self.ensureGrad();
        for (let i = 0; i < ga.length; i++) ga[i] += g[0];
      });
    }

    const ax = normalizeAxis(axis, this.ndim);
    const outShape = this.shape
      .map((d, i) => (i === ax ? 1 : d))
      .filter((_, i) => keepDims || i !== ax);
    // Precompute input-flat-index -> output-flat-index once; reused by backward.
    const inStrides = stridesFor(this.shape);
    const map = new Int32Array(this.size);
    {
      const reducedShape = this.shape.map((d, i) => (i === ax ? 1 : d));
      const reducedStrides = stridesFor(reducedShape);
      for (let i = 0; i < this.size; i++) {
        let rem = i;
        let outIdx = 0;
        for (let d = 0; d < this.ndim; d++) {
          const coord = Math.floor(rem / inStrides[d]);
          rem -= coord * inStrides[d];
          if (d !== ax) outIdx += coord * reducedStrides[d];
        }
        map[i] = outIdx;
      }
    }
    const data = new Float32Array(sizeOf(outShape));
    for (let i = 0; i < this.size; i++) data[map[i]] += this.data[i];
    const self = this;
    return Tensor.make(data, outShape, [this], (g) => {
      if (!self.requiresGrad) return;
      const ga = self.ensureGrad();
      for (let i = 0; i < ga.length; i++) ga[i] += g[map[i]];
    });
  }

  mean(axis: number | null = null, keepDims = false): Tensor {
    const count = axis === null ? this.size : this.shape[normalizeAxis(axis, this.ndim)];
    return this.sum(axis, keepDims).mulScalar(1 / count);
  }

  // ------------------------------------------------------------------ linalg

  /** 2-D matrix multiplication: [m,k] @ [k,n] -> [m,n]. */
  matmul(b: Tensor): Tensor {
    if (this.ndim !== 2 || b.ndim !== 2) {
      throw new Error(`matmul requires 2-D tensors, got ${this.ndim}-D and ${b.ndim}-D`);
    }
    const [m, k] = this.shape;
    const [k2, n] = b.shape;
    if (k !== k2) {
      throw new Error(`matmul shape mismatch: [${this.shape.join(",")}] @ [${b.shape.join(",")}]`);
    }
    const a = this;
    const data = new Float32Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let p = 0; p < k; p++) {
        const av = a.data[i * k + p];
        if (av === 0) continue;
        const rowOff = p * n;
        const outOff = i * n;
        for (let j = 0; j < n; j++) data[outOff + j] += av * b.data[rowOff + j];
      }
    }
    return Tensor.make(data, [m, n], [a, b], (g) => {
      if (a.requiresGrad) {
        const ga = a.ensureGrad(); // dA = g @ B^T
        for (let i = 0; i < m; i++) {
          for (let p = 0; p < k; p++) {
            let s = 0;
            for (let j = 0; j < n; j++) s += g[i * n + j] * b.data[p * n + j];
            ga[i * k + p] += s;
          }
        }
      }
      if (b.requiresGrad) {
        const gb = b.ensureGrad(); // dB = A^T @ g
        for (let p = 0; p < k; p++) {
          for (let j = 0; j < n; j++) {
            let s = 0;
            for (let i = 0; i < m; i++) s += a.data[i * k + p] * g[i * n + j];
            gb[p * n + j] += s;
          }
        }
      }
    });
  }

  /** 2-D transpose. */
  transpose(): Tensor {
    if (this.ndim !== 2) throw new Error(`transpose requires a 2-D tensor, got ${this.ndim}-D`);
    const [m, n] = this.shape;
    const a = this;
    const data = new Float32Array(this.size);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) data[j * m + i] = a.data[i * n + j];
    return Tensor.make(data, [n, m], [a], (g) => {
      if (!a.requiresGrad) return;
      const ga = a.ensureGrad();
      for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) ga[i * n + j] += g[j * m + i];
    });
  }

  /** Reshape to a same-size shape. Data is contiguous, so this is layout-preserving. */
  reshape(shape: Shape): Tensor {
    if (sizeOf(shape) !== this.size) {
      throw new Error(`Cannot reshape size ${this.size} to [${shape.join(",")}]`);
    }
    if (shapesEqual(shape, this.shape)) return this;
    const a = this;
    return Tensor.make(this.data.slice(), shape, [a], (g) => {
      if (!a.requiresGrad) return;
      const ga = a.ensureGrad();
      for (let i = 0; i < ga.length; i++) ga[i] += g[i];
    });
  }

  /** General N-D axis permutation. `perm[i]` is the source axis for output axis i. */
  permute(perm: number[]): Tensor {
    const nd = this.ndim;
    if (perm.length !== nd || new Set(perm).size !== nd || perm.some((p) => p < 0 || p >= nd)) {
      throw new Error(`permute expects a permutation of 0..${nd - 1}, got [${perm.join(",")}]`);
    }
    const inStrides = stridesFor(this.shape);
    const outShape = perm.map((p) => this.shape[p]);
    const outStrides = stridesFor(outShape);
    const n = this.size;
    const data = new Float32Array(n);
    // map[outFlat] = inFlat; reused by backward so the coordinate math runs once.
    const map = new Int32Array(n);
    for (let o = 0; o < n; o++) {
      let rem = o;
      let src = 0;
      for (let d = 0; d < nd; d++) {
        const c = Math.floor(rem / outStrides[d]);
        rem -= c * outStrides[d];
        src += c * inStrides[perm[d]];
      }
      map[o] = src;
      data[o] = this.data[src];
    }
    const a = this;
    return Tensor.make(data, outShape, [a], (g) => {
      if (!a.requiresGrad) return;
      const ga = a.ensureGrad();
      for (let o = 0; o < n; o++) ga[map[o]] += g[o];
    });
  }

  /** Batched 3-D matmul: [g,m,k] @ [g,k,n] -> [g,m,n]. */
  bmm(b: Tensor): Tensor {
    if (this.ndim !== 3 || b.ndim !== 3) {
      throw new Error(`bmm requires 3-D tensors, got ${this.ndim}-D and ${b.ndim}-D`);
    }
    const [g0, m, k] = this.shape;
    const [g1, k2, n] = b.shape;
    if (g0 !== g1 || k !== k2) {
      throw new Error(`bmm shape mismatch: [${this.shape.join(",")}] @ [${b.shape.join(",")}]`);
    }
    const a = this;
    const data = new Float32Array(g0 * m * n);
    for (let g = 0; g < g0; g++) {
      const aOff = g * m * k;
      const bOff = g * k * n;
      const oOff = g * m * n;
      for (let i = 0; i < m; i++) {
        for (let p = 0; p < k; p++) {
          const av = a.data[aOff + i * k + p];
          if (av === 0) continue;
          const rowOff = bOff + p * n;
          const outOff = oOff + i * n;
          for (let j = 0; j < n; j++) data[outOff + j] += av * b.data[rowOff + j];
        }
      }
    }
    return Tensor.make(data, [g0, m, n], [a, b], (grad) => {
      if (a.requiresGrad) {
        const ga = a.ensureGrad(); // dA = dO @ B^T
        for (let g = 0; g < g0; g++) {
          const aOff = g * m * k;
          const bOff = g * k * n;
          const oOff = g * m * n;
          for (let i = 0; i < m; i++) {
            for (let p = 0; p < k; p++) {
              let s = 0;
              for (let j = 0; j < n; j++) s += grad[oOff + i * n + j] * b.data[bOff + p * n + j];
              ga[aOff + i * k + p] += s;
            }
          }
        }
      }
      if (b.requiresGrad) {
        const gb = b.ensureGrad(); // dB = A^T @ dO
        for (let g = 0; g < g0; g++) {
          const aOff = g * m * k;
          const bOff = g * k * n;
          const oOff = g * m * n;
          for (let p = 0; p < k; p++) {
            for (let j = 0; j < n; j++) {
              let s = 0;
              for (let i = 0; i < m; i++) s += a.data[aOff + i * k + p] * grad[oOff + i * n + j];
              gb[bOff + p * n + j] += s;
            }
          }
        }
      }
    });
  }

  /**
   * Embedding lookup: gathers rows of a [vocab, dim] table by index,
   * producing [indices.length, dim]. Backward scatter-adds into the table.
   */
  gatherRows(indices: ArrayLike<number>): Tensor {
    if (this.ndim !== 2) throw new Error(`gatherRows requires a 2-D table, got ${this.ndim}-D`);
    const [vocab, dim] = this.shape;
    const n = indices.length;
    const data = new Float32Array(n * dim);
    for (let i = 0; i < n; i++) {
      const r = indices[i];
      if (r < 0 || r >= vocab) throw new Error(`gatherRows index ${r} out of range [0, ${vocab})`);
      for (let j = 0; j < dim; j++) data[i * dim + j] = this.data[r * dim + j];
    }
    const a = this;
    return Tensor.make(data, [n, dim], [a], (g) => {
      if (!a.requiresGrad) return;
      const ga = a.ensureGrad();
      for (let i = 0; i < n; i++) {
        const r = indices[i];
        for (let j = 0; j < dim; j++) ga[r * dim + j] += g[i * dim + j];
      }
    });
  }

  /** Softmax over the last dimension (max-subtracted for numerical stability). */
  softmax(): Tensor {
    const d = this.shape[this.ndim - 1];
    if (!d) throw new Error("softmax requires a non-empty last dimension");
    const rows = this.size / d;
    const data = new Float32Array(this.size);
    for (let r = 0; r < rows; r++) {
      const off = r * d;
      let max = -Infinity;
      for (let j = 0; j < d; j++) if (this.data[off + j] > max) max = this.data[off + j];
      let sum = 0;
      for (let j = 0; j < d; j++) {
        const e = Math.exp(this.data[off + j] - max);
        data[off + j] = e;
        sum += e;
      }
      for (let j = 0; j < d; j++) data[off + j] /= sum;
    }
    const a = this;
    return Tensor.make(data, this.shape, [a], (g) => {
      if (!a.requiresGrad) return;
      const ga = a.ensureGrad();
      // dx_i = y_i * (g_i - sum_j g_j * y_j)
      for (let r = 0; r < rows; r++) {
        const off = r * d;
        let dot = 0;
        for (let j = 0; j < d; j++) dot += g[off + j] * data[off + j];
        for (let j = 0; j < d; j++) ga[off + j] += data[off + j] * (g[off + j] - dot);
      }
    });
  }

  /**
   * Causal mask for attention scores shaped [..., t, t]: entries strictly above
   * the diagonal are replaced by `value` (a large negative number), so a
   * following softmax assigns them ~zero weight. Masked entries are constants,
   * so they receive no gradient.
   */
  maskedFillCausal(value = -1e9): Tensor {
    if (this.ndim < 2) throw new Error("maskedFillCausal requires at least 2 dimensions");
    const t = this.shape[this.ndim - 1];
    if (this.shape[this.ndim - 2] !== t) {
      throw new Error(`maskedFillCausal requires a square trailing block, got [${this.shape.join(",")}]`);
    }
    const groups = this.size / (t * t);
    const data = this.data.slice();
    for (let g = 0; g < groups; g++) {
      const off = g * t * t;
      for (let r = 0; r < t; r++) {
        for (let c = r + 1; c < t; c++) data[off + r * t + c] = value;
      }
    }
    const a = this;
    return Tensor.make(data, this.shape, [a], (grad) => {
      if (!a.requiresGrad) return;
      const ga = a.ensureGrad();
      for (let g = 0; g < groups; g++) {
        const off = g * t * t;
        for (let r = 0; r < t; r++) {
          for (let c = 0; c <= r; c++) ga[off + r * t + c] += grad[off + r * t + c];
        }
      }
    });
  }

  /** GELU (tanh approximation), the activation used by GPT-2. */
  gelu(): Tensor {
    const C = Math.sqrt(2 / Math.PI);
    return Tensor.unary(
      this,
      (x) => 0.5 * x * (1 + Math.tanh(C * (x + 0.044715 * x * x * x))),
      (x, _y, g) => {
        const inner = C * (x + 0.044715 * x * x * x);
        const t = Math.tanh(inner);
        const dInner = C * (1 + 3 * 0.044715 * x * x);
        return g * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * dInner);
      },
    );
  }

  /**
   * SiLU / swish: x * sigmoid(x). Smooth and non-monotonic, and unlike GELU it
   * is cheap enough to be worth fusing. The gate activation in SwiGLU.
   */
  silu(): Tensor {
    return Tensor.unary(
      this,
      (x) => x / (1 + Math.exp(-x)),
      // y = x*s, so dy/dx = s + x*s*(1-s) = s + y*(1-s).
      (x, y, g) => {
        const s = 1 / (1 + Math.exp(-x));
        return g * (s + y * (1 - s));
      },
    );
  }

  /**
   * Rotary position embedding over a [groups, time, headDim] tensor.
   *
   * Each adjacent dimension pair (2i, 2i+1) is rotated by an angle proportional
   * to its position, so a later dot product between a query and a key depends
   * only on their *relative* distance. That replaces the learned position table
   * with zero parameters, and — because nothing is looked up by absolute index —
   * the model degrades gracefully past its trained context instead of running
   * off the end of a table.
   *
   * The rotation is orthogonal, so the backward pass is just the inverse
   * rotation (transpose = rotate by -theta).
   */
  rope(base = 10000): Tensor {
    if (this.ndim !== 3) throw new Error(`rope expects a [groups, time, headDim] tensor, got ${this.ndim}-D`);
    const [groups, t, hd] = this.shape;
    if (hd % 2 !== 0) throw new Error(`rope requires an even headDim, got ${hd}`);
    const half = hd / 2;

    // cos/sin depend only on (position, pair), not on the data — computed once
    // and reused by every group and by the backward pass.
    const cos = new Float32Array(t * half);
    const sin = new Float32Array(t * half);
    for (let p = 0; p < t; p++) {
      for (let i = 0; i < half; i++) {
        const theta = p / Math.pow(base, (2 * i) / hd);
        cos[p * half + i] = Math.cos(theta);
        sin[p * half + i] = Math.sin(theta);
      }
    }

    const data = new Float32Array(this.size);
    for (let g = 0; g < groups; g++) {
      for (let p = 0; p < t; p++) {
        const off = (g * t + p) * hd;
        for (let i = 0; i < half; i++) {
          const c = cos[p * half + i];
          const s = sin[p * half + i];
          const x0 = this.data[off + 2 * i];
          const x1 = this.data[off + 2 * i + 1];
          data[off + 2 * i] = x0 * c - x1 * s;
          data[off + 2 * i + 1] = x0 * s + x1 * c;
        }
      }
    }

    return Tensor.make(data, this.shape, [this], (grad) => {
      if (!this.requiresGrad) return;
      const ga = this.ensureGrad();
      for (let g = 0; g < groups; g++) {
        for (let p = 0; p < t; p++) {
          const off = (g * t + p) * hd;
          for (let i = 0; i < half; i++) {
            const c = cos[p * half + i];
            const s = sin[p * half + i];
            const g0 = grad[off + 2 * i];
            const g1 = grad[off + 2 * i + 1];
            ga[off + 2 * i] += g0 * c + g1 * s;
            ga[off + 2 * i + 1] += -g0 * s + g1 * c;
          }
        }
      }
    });
  }

  /**
   * Fused softmax + cross-entropy over logits [n, vocab] against integer class
   * targets, returning the mean loss as a scalar. Fusing avoids materializing
   * softmax probabilities in the graph and keeps the backward pass exact:
   * dLogits = (softmax - onehot) / n.
   */
  crossEntropyLogits(targets: ArrayLike<number>): Tensor {
    if (this.ndim !== 2) throw new Error(`crossEntropyLogits requires 2-D logits, got ${this.ndim}-D`);
    const [n, vocab] = this.shape;
    if (targets.length !== n) {
      throw new Error(`Expected ${n} targets, got ${targets.length}`);
    }
    const probs = new Float32Array(this.size);
    let totalLoss = 0;
    for (let i = 0; i < n; i++) {
      const off = i * vocab;
      let max = -Infinity;
      for (let j = 0; j < vocab; j++) if (this.data[off + j] > max) max = this.data[off + j];
      let sum = 0;
      for (let j = 0; j < vocab; j++) {
        const e = Math.exp(this.data[off + j] - max);
        probs[off + j] = e;
        sum += e;
      }
      for (let j = 0; j < vocab; j++) probs[off + j] /= sum;
      const target = targets[i];
      if (target < 0 || target >= vocab) {
        throw new Error(`Target ${target} out of range [0, ${vocab})`);
      }
      // logsumexp form avoids taking log of an underflowed probability.
      totalLoss += max + Math.log(sum) - this.data[off + target];
    }
    const a = this;
    return Tensor.make(new Float32Array([totalLoss / n]), [], [a], (g) => {
      if (!a.requiresGrad) return;
      const ga = a.ensureGrad();
      const scale = g[0] / n;
      for (let i = 0; i < n; i++) {
        const off = i * vocab;
        for (let j = 0; j < vocab; j++) ga[off + j] += scale * probs[off + j];
        ga[off + targets[i]] -= scale;
      }
    });
  }
}
