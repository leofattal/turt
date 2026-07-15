export { Tensor } from "./tensor.js";
export {
  type Shape,
  sizeOf,
  stridesFor,
  shapesEqual,
  broadcastShapes,
  broadcastIndexer,
  normalizeAxis,
} from "./shape.js";
export { type Rng, mulberry32, gaussian, defaultRng } from "./random.js";
