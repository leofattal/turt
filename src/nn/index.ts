export { Module, Sequential } from "./module.js";
export { Linear } from "./linear.js";
export { ReLU, Tanh, Sigmoid } from "./activations.js";
export { LayerNorm } from "./layernorm.js";
export { mseLoss } from "./losses.js";

// Planned layers (see PRD "Neural Networks"): CNN, RNN/LSTM/GRU, transformer
// encoder/decoder, multi-head attention, positional/rotary embeddings, MoE.
// Each should extend Module and compose primitive Tensor ops so autodiff
// applies without hand-written backward passes.
