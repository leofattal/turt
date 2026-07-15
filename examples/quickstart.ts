/**
 * Turt quickstart — run with: pnpm example
 *
 * Walks through the four working subsystems: tensors + autodiff,
 * training a model, tokenization, and semantic memory.
 */
import {
  Adam,
  BPETokenizer,
  InMemoryVectorStore,
  Linear,
  ReLU,
  Sequential,
  Tensor,
  Trainer,
  mulberry32,
} from "../src/index.js";

// ── 1. Tensors and automatic differentiation ────────────────────────────────
console.log("── 1. Tensors & autodiff");

const a = Tensor.fromArray([1, 2, 3, 4], [2, 2], true); // requiresGrad = true
const b = Tensor.fromArray([0.5, -1, 2, 0], [2, 2], true);

const loss = a.matmul(b).tanh().sum(); // any composition of ops
loss.backward(); // gradients flow to every input

console.log("loss        =", loss.item().toFixed(4));
console.log("dloss/da    =", [...a.grad!].map((v) => v.toFixed(3)));

// ── 2. Train a neural network ───────────────────────────────────────────────
console.log("\n── 2. Train an MLP to fit y = x² (seeded, reproducible)");

const rng = mulberry32(42);
const xs = Array.from({ length: 128 }, () => rng() * 2 - 1);
const input = Tensor.fromArray(xs, [128, 1]);
const target = Tensor.fromArray(xs.map((x) => x * x), [128, 1]);

const model = new Sequential(new Linear(1, 32, { rng }), new ReLU(), new Linear(32, 1, { rng }));
const trainer = new Trainer(model, new Adam(model.parameters(), { lr: 0.01 }));

trainer.fit([{ input, target }], {
  epochs: 500,
  onEpochEnd: (epoch, meanLoss) => {
    if (epoch % 100 === 0) console.log(`epoch ${String(epoch).padStart(3)}  loss ${meanLoss.toFixed(6)}`);
    return meanLoss > 1e-5; // returning false stops early
  },
});

const probe = Tensor.fromArray([0.5], [1, 1]);
console.log(`model(0.5)  = ${model.forward(probe).item().toFixed(4)}  (true: 0.25)`);

// ── 3. Tokenization ─────────────────────────────────────────────────────────
console.log("\n── 3. Byte-level BPE tokenizer");

const tokenizer = new BPETokenizer();
tokenizer.train("the turtle trained the tiny transformer. ".repeat(20), 300);

const ids = tokenizer.encode("the tiny turtle");
console.log("encoded     =", ids);
console.log("decoded     =", JSON.stringify(tokenizer.decode(ids)));
console.log("vocab size  =", tokenizer.vocabSize);

// ── 4. Long-term memory (semantic search) ───────────────────────────────────
console.log("\n── 4. Vector memory");

const memory = new InMemoryVectorStore();
memory.add({ id: "fact-1", text: "Turtles are reptiles", embedding: [0.9, 0.1, 0.0] });
memory.add({ id: "fact-2", text: "Tensors hold numbers", embedding: [0.0, 0.2, 0.9] });
memory.add({ id: "fact-3", text: "Turtles can live 100 years", embedding: [0.8, 0.3, 0.1] });

const hits = memory.search([1, 0, 0], 2); // query embedding
for (const { record, score } of hits) {
  console.log(`score ${score.toFixed(3)}  ${record.text}`);
}
