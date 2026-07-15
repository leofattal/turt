import type { Tensor } from "../math/tensor.js";

/**
 * Clips the global L2 norm of all parameter gradients to `maxNorm`,
 * scaling them in place. Returns the pre-clip norm.
 */
export function clipGradNorm(params: Tensor[], maxNorm: number): number {
  let sumSq = 0;
  for (const p of params) {
    const g = p.grad;
    if (!g) continue;
    for (let i = 0; i < g.length; i++) sumSq += g[i] * g[i];
  }
  const norm = Math.sqrt(sumSq);
  if (norm > maxNorm && norm > 0) {
    const scale = maxNorm / norm;
    for (const p of params) {
      const g = p.grad;
      if (!g) continue;
      for (let i = 0; i < g.length; i++) g[i] *= scale;
    }
  }
  return norm;
}
