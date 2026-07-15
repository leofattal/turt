import type { Tensor } from "../math/tensor.js";

/**
 * Compute backend interface for the inference engine.
 *
 * The Tensor class in src/math is the CPU reference implementation. Faster
 * backends (WebGPU, optional CUDA via native bindings, WASM-SIMD) implement
 * this interface so the engine can dispatch per-device without callers
 * changing (see PRD "Inference Engine").
 */
export interface Backend {
  readonly name: string;
  /** Whether this backend can run in the current environment. */
  isAvailable(): boolean | Promise<boolean>;
  matmul(a: Tensor, b: Tensor): Tensor | Promise<Tensor>;
}

/** Reference backend: delegates to the pure-TypeScript math engine. */
export class CpuBackend implements Backend {
  readonly name = "cpu";

  isAvailable(): boolean {
    return true;
  }

  matmul(a: Tensor, b: Tensor): Tensor {
    return a.matmul(b);
  }
}
