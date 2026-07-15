/**
 * Validates every transformer GPU op against the trusted CPU `Tensor` engine.
 *
 *   deno run --unstable-webgpu --unstable-sloppy-imports --allow-read \
 *     scripts/gpu-ops-validate.ts
 *
 * For each op, the same inputs run through both engines; a scalar loss
 * `sum(out * R)` is backpropagated and the input/parameter gradients are
 * compared to f32 tolerance. Exits non-zero on any mismatch.
 */

import { Tensor } from "../src/math/tensor.ts";
import { LayerNorm } from "../src/nn/layernorm.ts";
import { mulberry32, gaussian } from "../src/math/random.ts";
import { GpuEngine } from "../src/infer/webgpu/autograd.ts";

function randn(n: number, seed: number, std = 1): Float32Array {
  const rng = mulberry32(seed);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = gaussian(rng) * std;
  return out;
}
function maxRelErr(a: Float32Array, b: ArrayLike<number>): number {
  let worst = 0;
  for (let i = 0; i < a.length; i++) {
    const denom = Math.max(1e-3, Math.abs(a[i]), Math.abs(b[i]));
    worst = Math.max(worst, Math.abs(a[i] - b[i]) / denom);
  }
  return worst;
}
const TOL = 3e-3;
let failed = false;
function check(label: string, gpu: Float32Array, cpu: ArrayLike<number>): void {
  const err = maxRelErr(gpu, cpu);
  const ok = err <= TOL && Number.isFinite(err);
  console.log(`  ${ok ? "ok  " : "FAIL"} ${label.padEnd(26)} max rel err ${err.toExponential(2)}`);
  if (!ok) failed = true;
}

async function main(): Promise<void> {
  const e = await GpuEngine.create();
  console.log("Transformer GPU ops — validation vs CPU engine\n");

  // 1. gatherRows (embedding) ------------------------------------------------
  {
    const V = 20, D = 12, N = 16;
    const tableData = randn(V * D, 1, 0.3);
    const idx = Array.from({ length: N }, (_, i) => (i * 7 + 3) % V);
    const rData = randn(N * D, 2);
    const gt = e.tensor(tableData, [V, D], true);
    const gOut = gt.gatherRows(Uint32Array.from(idx)).mul(e.tensor(rData, [N, D])).sum();
    gOut.backward();
    const ct = new Tensor(tableData.slice(), [V, D], true);
    ct.gatherRows(idx).mul(new Tensor(rData.slice(), [N, D])).sum().backward();
    check("gatherRows dTable", await e.read(gt.grad!, gt.size), ct.grad!);
  }

  // 2. bmm -------------------------------------------------------------------
  {
    const G = 4, M = 8, K = 6, N = 10;
    const aD = randn(G * M * K, 3), bD = randn(G * K * N, 4), rD = randn(G * M * N, 5);
    const ga = e.tensor(aD, [G, M, K], true), gb = e.tensor(bD, [G, K, N], true);
    ga.bmm(gb).mul(e.tensor(rD, [G, M, N])).sum().backward();
    const ca = new Tensor(aD.slice(), [G, M, K], true), cb = new Tensor(bD.slice(), [G, K, N], true);
    ca.bmm(cb).mul(new Tensor(rD.slice(), [G, M, N])).sum().backward();
    check("bmm dA", await e.read(ga.grad!, ga.size), ca.grad!);
    check("bmm dB", await e.read(gb.grad!, gb.size), cb.grad!);
  }

  // 3. softmax ---------------------------------------------------------------
  {
    const rows = 12, d = 9;
    const xD = randn(rows * d, 6), rD = randn(rows * d, 7);
    const gx = e.tensor(xD, [rows, d], true);
    gx.softmax().mul(e.tensor(rD, [rows, d])).sum().backward();
    const cx = new Tensor(xD.slice(), [rows, d], true);
    cx.softmax().mul(new Tensor(rD.slice(), [rows, d])).sum().backward();
    check("softmax dX", await e.read(gx.grad!, gx.size), cx.grad!);
  }

  // 4. maskedFillCausal ------------------------------------------------------
  {
    const G = 2, T = 5;
    const xD = randn(G * T * T, 8), rD = randn(G * T * T, 9);
    const gx = e.tensor(xD, [G, T, T], true);
    gx.maskedFillCausal().softmax().mul(e.tensor(rD, [G, T, T])).sum().backward();
    const cx = new Tensor(xD.slice(), [G, T, T], true);
    cx.maskedFillCausal().softmax().mul(new Tensor(rD.slice(), [G, T, T])).sum().backward();
    check("maskedFillCausal dX", await e.read(gx.grad!, gx.size), cx.grad!);
  }

  // 5. gelu ------------------------------------------------------------------
  {
    const n = 40;
    const xD = randn(n, 10), rD = randn(n, 11);
    const gx = e.tensor(xD, [n], true);
    gx.gelu().mul(e.tensor(rD, [n])).sum().backward();
    const cx = new Tensor(xD.slice(), [n], true);
    cx.gelu().mul(new Tensor(rD.slice(), [n])).sum().backward();
    check("gelu dX", await e.read(gx.grad!, gx.size), cx.grad!);
  }

  // 6. permute (4-D) ---------------------------------------------------------
  {
    const s = [2, 3, 4, 5];
    const n = 2 * 3 * 4 * 5;
    const xD = randn(n, 12), rD = randn(n, 13);
    const gx = e.tensor(xD, s, true);
    gx.permute([0, 2, 1, 3]).mul(e.tensor(rD, [2, 4, 3, 5])).sum().backward();
    const cx = new Tensor(xD.slice(), s, true);
    cx.permute([0, 2, 1, 3]).mul(new Tensor(rD.slice(), [2, 4, 3, 5])).sum().backward();
    check("permute dX", await e.read(gx.grad!, gx.size), cx.grad!);
  }

  // 7. matmulBT (weight-tied head) ------------------------------------------
  {
    const N = 8, D = 6, V = 11;
    const xD = randn(N * D, 14), wD = randn(V * D, 15, 0.2), rD = randn(N * V, 16);
    const gx = e.tensor(xD, [N, D], true), gw = e.tensor(wD, [V, D], true);
    gx.matmulBT(gw).mul(e.tensor(rD, [N, V])).sum().backward();
    const cx = new Tensor(xD.slice(), [N, D], true), cw = new Tensor(wD.slice(), [V, D], true);
    cx.matmul(cw.transpose()).mul(new Tensor(rD.slice(), [N, V])).sum().backward();
    check("matmulBT dX", await e.read(gx.grad!, gx.size), cx.grad!);
    check("matmulBT dW", await e.read(gw.grad!, gw.size), cw.grad!);
  }

  // 8. layerNorm -------------------------------------------------------------
  {
    const rows = 10, d = 8;
    const xD = randn(rows * d, 17), rD = randn(rows * d, 18);
    const gammaD = randn(d, 19, 0.5).map((v) => v + 1);
    const betaD = randn(d, 20, 0.5);
    const gx = e.tensor(xD, [rows, d], true);
    const gGamma = e.tensor(gammaD, [d], true), gBeta = e.tensor(betaD, [d], true);
    gx.layerNorm(gGamma, gBeta).mul(e.tensor(rD, [rows, d])).sum().backward();

    const cx = new Tensor(xD.slice(), [rows, d], true);
    const ln = new LayerNorm(d);
    ln.gamma.data.set(gammaD);
    ln.beta.data.set(betaD);
    ln.forward(cx).mul(new Tensor(rD.slice(), [rows, d])).sum().backward();
    check("layerNorm dX", await e.read(gx.grad!, gx.size), cx.grad!);
    check("layerNorm dGamma", await e.read(gGamma.grad!, d), ln.gamma.grad!);
    check("layerNorm dBeta", await e.read(gBeta.grad!, d), ln.beta.grad!);
  }

  // 9. crossEntropyLogits (fused loss) --------------------------------------
  {
    const N = 14, V = 17;
    const xD = randn(N * V, 21, 0.5);
    const tgt = Array.from({ length: N }, (_, i) => (i * 5 + 2) % V);
    const gx = e.tensor(xD, [N, V], true);
    const gLoss = gx.crossEntropyLogits(Uint32Array.from(tgt));
    gLoss.backward();
    const cx = new Tensor(xD.slice(), [N, V], true);
    const cLoss = cx.crossEntropyLogits(tgt);
    cLoss.backward();
    check("crossEntropy loss", await e.read(gLoss.buffer, 1), [cLoss.item()]);
    check("crossEntropy dLogits", await e.read(gx.grad!, gx.size), cx.grad!);
  }

  if (failed) {
    console.error("\nVALIDATION FAILED");
    Deno.exit(1);
  }
  console.log("\nAll transformer GPU ops match the CPU engine within tolerance.");
}

main().catch((err) => {
  console.error(err);
  Deno.exit(1);
});
