import { Tensor } from "../math/tensor.js";
import { Module } from "./module.js";

export class ReLU extends Module {
  forward(input: Tensor): Tensor {
    return input.relu();
  }
}

export class Tanh extends Module {
  forward(input: Tensor): Tensor {
    return input.tanh();
  }
}

export class Sigmoid extends Module {
  forward(input: Tensor): Tensor {
    return input.sigmoid();
  }
}
