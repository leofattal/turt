/**
 * Data-parallel training worker.
 *
 * Each worker owns a full replica of the model. Per step it receives the
 * current parameters plus a micro-batch, runs forward and backward, and sends
 * the gradients back. The main thread averages gradients across workers and
 * takes the optimizer step, so the result is mathematically identical to a
 * single-threaded run over the full batch — just spread across cores.
 *
 * Gradient buffers are transferred (not copied) back to the main thread.
 */

import { parentPort, workerData } from "node:worker_threads";
import { GPT, type GPTConfig } from "../src/models/gpt.js";
import { Tensor } from "../src/math/tensor.js";
import { mulberry32 } from "../src/math/random.js";
import { flattenGrads, loadParams } from "./params.js";

export interface WorkerInit {
  config: GPTConfig;
  seed: number;
}

export interface StepRequest {
  params: Float32Array;
  inputs: Float32Array;
  targets: Int32Array;
  batchSize: number;
  blockSize: number;
}

export interface StepResponse {
  grads: Float32Array;
  loss: number;
}

const { config, seed } = workerData as WorkerInit;
// The seed only affects initial weights, which the main thread immediately
// overwrites with its own — replicas stay identical regardless.
const model = new GPT(config, mulberry32(seed));
const params = model.parameters();

parentPort!.on("message", (request: StepRequest) => {
  loadParams(params, request.params);
  for (const p of params) p.zeroGrad();

  const idx = new Tensor(request.inputs, [request.batchSize, request.blockSize]);
  const loss = model.loss(idx, request.targets);
  loss.backward();

  const grads = flattenGrads(params);
  const response: StepResponse = { grads, loss: loss.item() };
  parentPort!.postMessage(response, [grads.buffer]);
});
