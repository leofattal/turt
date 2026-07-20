# Turt the Chatbot

Turt can be turned into a small conversational chatbot by **instruction-tuning**
the pretrained GPT — the same two-stage recipe (pretrain, then fine-tune on
conversations) that real assistants use, at a scale that runs on a laptop.

```bash
pnpm prepare-chat     # build the synthetic chat dataset
pnpm finetune-chat    # instruction-tune on the GPU (~20 min, needs Deno)
pnpm chat             # talk to it in the terminal
pnpm serve            # or open the browser chat UI at http://localhost:8787
```

## How it works

1. **Chat data** (`scripts/prepare-chat.ts`). A synthetic corpus of `User:` /
   `Turt:` dialogues with a consistent persona. Each intent (greetings, "what
   are you", capabilities, thanks, farewells, small talk, and a graceful
   fallback for out-of-scope questions) has many paraphrased user messages and
   several reply variants, composed into short multi-turn conversations. It is
   encoded with the **same** BPE tokenizer as pretraining — it must be, since
   the model's embeddings are indexed by those exact token ids. A little raw
   novel text is mixed in to keep general English fluent.

2. **Instruction tuning** (`scripts/finetune-chat.ts`). The pretrained novel
   checkpoint is loaded into the GPU-resident `GpuGPT`, then fine-tuned on the
   chat data with a low learning rate. This is where the model learns the
   turn-taking format and the persona while keeping the language ability it
   learned from books. Running on the GPU makes this ~50× faster than the CPU
   trainer, so it finishes in minutes. Checkpoints are saved in CPU format
   (`checkpoints/chat.bin`) by reading the GPU weights back into a CPU model.

3. **Chat CLI** (`scripts/chat.ts`). Maintains a short conversation history
   (bounded by the model's context window), formats each turn in the tuned
   template, and **streams** the reply token by token, stopping generation at
   the newline that ends the assistant's turn so it doesn't hallucinate your
   next message (and doesn't waste compute past the stop). In-chat commands:
   `/reset` clears history, `/temp <x>` and `/topk <n>` adjust sampling.

   The CLI and server pick the tokenizer that matches the checkpoint's vocab:
   `data/tokenizer.json` if it matches, else `data/tokenizer-<vocab>.json`.
   This matters because rebuilding the data pipeline with a bigger vocab (as
   the gpt-big run did) replaces `tokenizer.json` and would otherwise orphan
   every older checkpoint — the ids stop matching the embeddings the model
   learned. The 1024-vocab tokenizer the current `chat.bin` needs lives at
   `data/tokenizer-1024.json` (regenerated deterministically from the cached
   Gutenberg books with the original prepare-data settings).

4. **Web chat** (`scripts/serve.ts`). The same reply logic wrapped in a
   zero-dependency Node HTTP server: it loads the tuned checkpoint once, serves
   a self-contained browser chat UI at `/`, and answers `POST /api/chat`
   (`{history, message} -> {reply}`). The browser holds the conversation
   history and sends it with each turn, so multi-turn context works exactly as
   in the CLI. Run `pnpm serve` and open http://localhost:8787.

## What to expect — honestly

This is a **real chatbot in mechanism** — two-stage training, turn-taking, a chat
template, bounded conversational memory — and a **limited one in ability**, capped
by the 2M-parameter model:

- **It handles well**: greetings, questions about itself, capabilities, thanks,
  farewells, and simple small talk — the patterns it was tuned on.
- **It handles gracefully**: out-of-scope questions get a "I'm a small model, I
  can't really help with that" style fallback, because that was in the training
  data.
- **It does not**: reason, do math, recall facts, follow complex instructions,
  or stay coherent on anything far from its training distribution. It has ~128
  tokens of memory, so it forgets earlier turns quickly.

It is a demonstration that the full assistant-building pipeline works end to end
from scratch — not a useful assistant. Crossing that gap needs the two things
covered in [GPU.md](./GPU.md): a much larger model (cloud GPU) and a richer
instruction-tuning dataset.
