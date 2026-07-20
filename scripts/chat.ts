/**
 * Interactive chat with the fine-tuned "Turt" chatbot.
 *
 *   pnpm chat [--temperature 0.7] [--top-k 40] [--max-tokens 40] [--ckpt checkpoints/chat.bin]
 *            [--tokenizer data/tokenizer.json]
 *
 * The tokenizer is chosen to match the checkpoint's vocab (see
 * loadMatchingTokenizer) — a rebuilt data/tokenizer.json with a different
 * vocab would otherwise scramble or crash an older model.
 *
 * Maintains a short conversation history (bounded by the model's context
 * window), formats each turn in the User:/Turt: template the model was tuned
 * on, and streams the reply token by token, stopping at the newline that ends
 * the assistant's turn so it doesn't hallucinate your next message.
 *
 * In-chat commands: /reset clears the history, /temp <x> and /topk <n> adjust
 * sampling, /help lists these. Type 'exit' or Ctrl-D to quit.
 */

import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { createInterface } from "node:readline";
import { GPT } from "../src/models/gpt.js";
import { mulberry32 } from "../src/math/random.js";
import { loadCheckpoint, loadMatchingTokenizer, type CheckpointMeta } from "./checkpoint.js";

const ROOT = join(import.meta.dirname, "..");

const flag = (name: string, fallback: string): string => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
};

// Default to the biggest chat model available: the 13.77M chat-big.bin once its
// pretrain + fine-tune lands, falling back to the original 2M chat.bin.
const defaultCkpt = (): string =>
  existsSync(join(ROOT, "checkpoints/chat-big.bin"))
    ? "checkpoints/chat-big.bin"
    : "checkpoints/chat.bin";

async function main(): Promise<void> {
  const ckptName = flag("ckpt", defaultCkpt());
  const ckptPath = join(ROOT, ckptName);
  let temperature = Number(flag("temperature", "0.7"));
  let topK = Number(flag("top-k", "40"));
  const maxNewTokens = Number(flag("max-tokens", "40"));

  const buf = await readFile(ckptPath);
  const headerLen = buf.readUInt32LE(8);
  const meta = JSON.parse(buf.subarray(12, 12 + headerLen).toString("utf8")) as CheckpointMeta;
  const tokenizer = await loadMatchingTokenizer(
    join(ROOT, "data"),
    meta.config.vocabSize,
    flag("tokenizer", "") || undefined,
  );
  const model = new GPT(meta.config, mulberry32(1));
  await loadCheckpoint(ckptPath, model);
  const blockSize = meta.config.blockSize;
  const rng = mulberry32(Date.now() % 100000);

  let history: Array<{ user: string; turt: string }> = [];

  /** Builds the prompt, trimming oldest turns so it fits the context window. */
  const buildPrompt = (userInput: string): string => {
    let start = 0;
    for (;;) {
      const convo =
        history.slice(start).map((e) => `User: ${e.user}\nTurt: ${e.turt}\n`).join("") +
        `User: ${userInput}\nTurt:`;
      if (tokenizer.encode(convo).length <= blockSize - maxNewTokens || start >= history.length) {
        return convo;
      }
      start++;
    }
  };

  /**
   * Generates the reply, streaming it to stdout as tokens are sampled.
   * Cumulative decoding handles BPE tokens that split multi-byte characters:
   * text ending in an incomplete sequence (U+FFFD) is held back until the next
   * token completes it. Generation stops at the newline that ends the turn.
   */
  const reply = (userInput: string): string => {
    const prompt = buildPrompt(userInput);
    const promptIds = tokenizer.encode(prompt);
    const genIds: number[] = [];
    let printed = "";
    model.generate(promptIds, maxNewTokens, {
      temperature,
      topK,
      rng,
      onToken: (id) => {
        genIds.push(id);
        let text = tokenizer.decode(genIds);
        const nl = text.indexOf("\n");
        const done = nl >= 0;
        if (done) text = text.slice(0, nl);
        else if (text.endsWith("�")) return; // wait for the rest of the character
        text = text.replace(/^\s+/, ""); // the tuned template puts a space after "Turt:"
        process.stdout.write(text.slice(printed.length));
        printed = text;
        if (done) return false;
      },
    });
    process.stdout.write(printed ? "\n\n" : "…\n\n");
    return printed.trim() || "…";
  };

  console.log(
    `\nTurt — a tiny from-scratch chatbot. Type 'exit' to quit, /help for commands.`,
  );
  console.log(
    `Model: ${meta.config.nLayer}L/${meta.config.nHead}H/${meta.config.nEmbd}D, ` +
      `${(model.numParams() / 1e6).toFixed(2)}M params, ${blockSize}-token context — ${ckptName}`,
  );
  console.log("It's best at greetings, questions about itself, and small talk.\n");
  console.log("Turt: Hello! I'm Turt. What would you like to talk about?\n");

  const rl = createInterface({ input: process.stdin, output: process.stdout, prompt: "You: " });
  rl.prompt();
  rl.on("line", (line) => {
    const input = line.trim();
    if (!input) return rl.prompt();
    if (["exit", "quit", "bye"].includes(input.toLowerCase())) {
      console.log("Turt: Goodbye! Take care.\n");
      return rl.close();
    }
    if (input.startsWith("/")) {
      const [cmd, arg] = input.slice(1).split(/\s+/);
      switch (cmd) {
        case "reset":
          history = [];
          console.log("(history cleared)\n");
          break;
        case "temp":
          if (arg && Number(arg) > 0) temperature = Number(arg);
          console.log(`(temperature ${temperature})\n`);
          break;
        case "topk":
          if (arg && Number(arg) > 0) topK = Number(arg);
          console.log(`(top-k ${topK})\n`);
          break;
        case "help":
          console.log("(/reset clears history, /temp <x> sets temperature, /topk <n> sets top-k)\n");
          break;
        default:
          console.log(`(unknown command /${cmd} — try /help)\n`);
      }
      return rl.prompt();
    }
    process.stdout.write("Turt: ");
    const answer = reply(input);
    history.push({ user: input, turt: answer });
    rl.prompt();
  });
  rl.on("close", () => process.exit(0));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
