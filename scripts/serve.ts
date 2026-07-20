/**
 * Web chat server for the fine-tuned "Turt" chatbot.
 *
 *   pnpm serve [--port 8787] [--temperature 0.7] [--top-k 40] [--ckpt checkpoints/chat.bin]
 *
 * Loads the tuned GPT once at startup and serves two things over plain HTTP
 * (no framework, no dependencies beyond the Turt engine):
 *
 *   GET  /            — a self-contained browser chat UI
 *   POST /api/chat    — { history: [{user, turt}], message } -> { reply }
 *
 * The reply logic mirrors scripts/chat.ts exactly: history is formatted in the
 * User:/Turt: template the model was tuned on, trimmed to fit the 128-token
 * context, generated, and cut at the newline that ends the assistant's turn.
 */

import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { join } from "node:path";
import { GPT } from "../src/models/gpt.js";
import { mulberry32 } from "../src/math/random.js";
import { loadCheckpoint, loadMatchingTokenizer, type CheckpointMeta } from "./checkpoint.js";

const ROOT = join(import.meta.dirname, "..");

const flag = (name: string, fallback: string): string => {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
};

const MAX_NEW_TOKENS = 40;

interface Turn {
  user: string;
  turt: string;
}

// Same default as chat.ts: prefer the 13.77M chat-big.bin once it exists.
const defaultCkpt = (): string =>
  existsSync(join(ROOT, "checkpoints/chat-big.bin"))
    ? "checkpoints/chat-big.bin"
    : "checkpoints/chat.bin";

async function main(): Promise<void> {
  const port = Number(flag("port", "8787"));
  const ckptName = flag("ckpt", defaultCkpt());
  const ckptPath = join(ROOT, ckptName);
  const temperature = Number(flag("temperature", "0.7"));
  const topK = Number(flag("top-k", "40"));

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
  const rng = mulberry32(1234);

  /** Builds the prompt, trimming oldest turns so it fits the context window. */
  const buildPrompt = (history: Turn[], userInput: string): string => {
    let start = 0;
    for (;;) {
      const convo =
        history.slice(start).map((e) => `User: ${e.user}\nTurt: ${e.turt}\n`).join("") +
        `User: ${userInput}\nTurt:`;
      if (tokenizer.encode(convo).length <= blockSize - MAX_NEW_TOKENS || start >= history.length) {
        return convo;
      }
      start++;
    }
  };

  const reply = (history: Turn[], userInput: string): string => {
    const prompt = buildPrompt(history, userInput);
    const promptIds = tokenizer.encode(prompt);
    const gen = model.generate(promptIds, MAX_NEW_TOKENS, { temperature, topK, rng });
    const full = tokenizer.decode(gen);
    const answer = full.slice(prompt.length).split("\n")[0].trim();
    return answer || "…";
  };

  const server = createServer((req, res) => {
    if (req.method === "GET" && (req.url === "/" || req.url === "/index.html")) {
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(PAGE);
      return;
    }

    if (req.method === "POST" && req.url === "/api/chat") {
      let body = "";
      req.on("data", (chunk) => {
        body += chunk;
        if (body.length > 1_000_000) req.destroy(); // guard against runaway payloads
      });
      req.on("end", () => {
        try {
          const parsed = JSON.parse(body || "{}") as { history?: Turn[]; message?: string };
          const message = (parsed.message ?? "").toString().slice(0, 2000).trim();
          const history = Array.isArray(parsed.history) ? parsed.history.slice(-20) : [];
          if (!message) {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: "empty message" }));
            return;
          }
          const answer = reply(history, message);
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ reply: answer }));
        } catch (err) {
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: String(err) }));
        }
      });
      return;
    }

    res.writeHead(404, { "Content-Type": "text/plain" });
    res.end("Not found");
  });

  server.listen(port, () => {
    console.log(`\nTurt web chat running at http://localhost:${port}`);
    console.log(`Model: ${meta.config.nLayer}L/${meta.config.nHead}H/${meta.config.nEmbd}D, ` +
      `${(model.numParams() / 1e6).toFixed(2)}M params — checkpoint ${ckptName}`);
    console.log("Press Ctrl-C to stop.\n");
  });
}

const PAGE = /* html */ `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Turt — tiny from-scratch chatbot</title>
<style>
  :root {
    --bg: #0f1115; --panel: #171a21; --border: #262b36; --text: #e6e9ef;
    --muted: #8b93a3; --user: #2b6cff; --turt: #1f2430; --accent: #4ade80;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); height: 100vh; display: flex; flex-direction: column;
  }
  header {
    padding: 14px 18px; border-bottom: 1px solid var(--border); display: flex;
    align-items: baseline; gap: 10px;
  }
  header h1 { margin: 0; font-size: 16px; font-weight: 600; }
  header .tag { color: var(--muted); font-size: 12px; }
  header .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); }
  main {
    flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column;
    gap: 12px; max-width: 760px; width: 100%; margin: 0 auto;
  }
  .msg { display: flex; gap: 10px; max-width: 88%; }
  .msg.user { align-self: flex-end; flex-direction: row-reverse; }
  .bubble {
    padding: 9px 13px; border-radius: 14px; white-space: pre-wrap; word-wrap: break-word;
  }
  .msg.user .bubble { background: var(--user); color: #fff; border-bottom-right-radius: 4px; }
  .msg.turt .bubble { background: var(--turt); border: 1px solid var(--border); border-bottom-left-radius: 4px; }
  .who { font-size: 11px; color: var(--muted); align-self: flex-end; padding-bottom: 3px; }
  .typing .bubble { color: var(--muted); font-style: italic; }
  footer {
    border-top: 1px solid var(--border); padding: 14px; display: flex; gap: 10px;
    max-width: 760px; width: 100%; margin: 0 auto;
  }
  #input {
    flex: 1; background: var(--panel); border: 1px solid var(--border); color: var(--text);
    border-radius: 10px; padding: 11px 14px; font: inherit; resize: none; outline: none;
  }
  #input:focus { border-color: var(--user); }
  #send {
    background: var(--user); color: #fff; border: none; border-radius: 10px; padding: 0 20px;
    font: inherit; font-weight: 600; cursor: pointer;
  }
  #send:disabled { opacity: .5; cursor: default; }
  .hint { max-width: 760px; margin: 0 auto; padding: 0 14px 10px; color: var(--muted); font-size: 12px; }
</style>
</head>
<body>
  <header>
    <span class="dot"></span>
    <h1>Turt</h1>
    <span class="tag">2M-param chatbot, trained from scratch</span>
  </header>
  <main id="log">
    <div class="msg turt"><div class="bubble">Hello! I'm Turt. What would you like to talk about?</div></div>
  </main>
  <div class="hint">Best at greetings, questions about itself, capabilities, and small talk. It has ~128 tokens of memory.</div>
  <footer>
    <textarea id="input" rows="1" placeholder="Say something to Turt…" autofocus></textarea>
    <button id="send">Send</button>
  </footer>
<script>
  const log = document.getElementById("log");
  const input = document.getElementById("input");
  const send = document.getElementById("send");
  const history = [];

  function bubble(text, who) {
    const msg = document.createElement("div");
    msg.className = "msg " + who;
    const b = document.createElement("div");
    b.className = "bubble";
    b.textContent = text;
    msg.appendChild(b);
    log.appendChild(msg);
    log.scrollTop = log.scrollHeight;
    return msg;
  }

  async function submit() {
    const message = input.value.trim();
    if (!message) return;
    input.value = "";
    input.style.height = "auto";
    bubble(message, "user");
    send.disabled = true;
    const typing = bubble("Turt is thinking…", "turt");
    typing.classList.add("typing");
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ history, message }),
      });
      const data = await res.json();
      const reply = data.reply || "…";
      typing.remove();
      bubble(reply, "turt");
      history.push({ user: message, turt: reply });
    } catch (e) {
      typing.remove();
      bubble("(error reaching the model)", "turt");
    } finally {
      send.disabled = false;
      input.focus();
    }
  }

  send.addEventListener("click", submit);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 140) + "px";
  });
</script>
</body>
</html>`;

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
