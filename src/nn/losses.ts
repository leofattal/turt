import { Tensor } from "../math/tensor.js";

/** Mean squared error over all elements. */
export function mseLoss(prediction: Tensor, target: Tensor): Tensor {
  const diff = prediction.sub(target);
  return diff.mul(diff).mean();
}
