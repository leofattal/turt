/**
 * Builds a billion-token-class training corpus by streaming C4 (the cleaned
 * Common Crawl web-text dataset, allenai/c4 "en") straight into tokenized
 * train/val bins. This is the scale-up counterpart to prepare-data.ts: that
 * script holds the whole corpus in memory, which is fine at 250MB of Gutenberg
 * and impossible at the ~8GB of text a GPT-2-class model wants. This one holds
 * only one document at a time.
 *
 * Why C4: it ships as ~1024 gzipped JSON-lines shards downloadable over plain
 * HTTPS with no auth — so the from-scratch pipeline can consume it with fetch +
 * DecompressionStream, no parquet parser or hf client. It is also the kind of
 * varied modern prose (explanations, articles, how-tos) that the Phi result
 * says small models spend capacity on far better than archaic fiction.
 *
 * Everything is written under --dir (default data-big/), NOT data/, so the
 * small-corpus artifacts a local run may currently be using are never touched.
 * Point the trainer at it with: pnpm gpu-pretrain --data data-big ...
 *
 * Sizing: default target is 8 GB of raw text ≈ ~1.9B tokens at ~4.2 chars/tok.
 * That is slightly under Chinchilla-optimal for 124M params (20 tok/param =
 * 2.5B) but keeps train.bin under 4GB, the safe typed-array ceiling in V8.
 *
 * Run: pnpm prepare-data-big [--target-gb 8] [--vocab 16384] [--dir data-big]
 *      [--sample-mb 48] [--val-every 20]
 *
 * Outputs (under --dir, gitignored):
 *   raw/c4-train.*.json.gz  cached shard downloads (re-runs cost nothing)
 *   tokenizer.json          learned BPE merges (reused if already present)
 *   train.bin / val.bin     Uint16 token ids (val = every --val-every'th doc)
 *   meta.json               vocab size, token counts, shard list
 */

import { createReadStream, createWriteStream, type WriteStream } from "node:fs";
import { mkdir, readFile, rename, stat, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { Readable } from "node:stream";
import { BPETokenizer } from "../src/tokenizer/bpe.js";

const SHARD_COUNT = 1024;
const shardName = (i: number): string => `c4-train.${String(i).padStart(5, "0")}-of-01024.json.gz`;
const shardUrl = (i: number): string =>
  `https://huggingface.co/datasets/allenai/c4/resolve/main/en/${shardName(i)}`;

function parseNum(name: string, fallback: number): number {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? Number(process.argv[i + 1]) : fallback;
}
function parseStr(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

/** Same spirit as prepare-data.ts's gate, but light: C4 is already cleaned. */
function looksLikeEnglish(text: string): boolean {
  if (text.length < 200) return false;
  const sample = text.slice(0, 4096);
  let ascii = 0;
  for (let i = 0; i < sample.length; i++) if (sample.charCodeAt(i) < 128) ascii++;
  return ascii / sample.length > 0.9;
}

/** Downloads one shard to the cache (atomically, via a temp name) if missing. */
async function ensureShard(rawDir: string, index: number): Promise<string> {
  const path = join(rawDir, shardName(index));
  try {
    await stat(path);
    return path;
  } catch {
    // not cached yet
  }
  const response = await fetch(shardUrl(index), {
    headers: { "User-Agent": "turt-research/0.1 (educational LLM training)" },
  });
  if (!response.ok || !response.body) throw new Error(`shard ${index}: HTTP ${response.status}`);
  const tmp = `${path}.part`;
  const out = createWriteStream(tmp);
  let bytes = 0;
  for await (const chunk of response.body) {
    bytes += chunk.length;
    if (!out.write(chunk)) await new Promise((r) => out.once("drain", r));
    process.stdout.write(`\r  downloading ${shardName(index)}  ${(bytes / 1024 / 1024).toFixed(0)} MB   `);
  }
  await new Promise((resolve, reject) => out.end(() => resolve(null)).once("error", reject));
  await rename(tmp, path);
  process.stdout.write("\n");
  return path;
}

/** Streams parsed documents out of shard files in order, one doc in memory at a time. */
async function* docs(rawDir: string): AsyncGenerator<string> {
  for (let i = 0; i < SHARD_COUNT; i++) {
    const path = await ensureShard(rawDir, i);
    const lines = Readable.toWeb(createReadStream(path))
      .pipeThrough(new DecompressionStream("gzip"))
      .pipeThrough(new TextDecoderStream());
    let tail = "";
    for await (const chunk of lines) {
      const parts = (tail + chunk).split("\n");
      tail = parts.pop()!;
      for (const line of parts) {
        if (!line) continue;
        const text = (JSON.parse(line) as { text: string }).text;
        if (looksLikeEnglish(text)) yield text;
      }
    }
    if (tail) {
      const text = (JSON.parse(tail) as { text: string }).text;
      if (looksLikeEnglish(text)) yield text;
    }
  }
}

/** Loads the tokenizer if present; otherwise trains one on the leading docs. */
async function ensureTokenizer(dir: string, rawDir: string, vocabSize: number, sampleBytes: number): Promise<BPETokenizer> {
  const path = join(dir, "tokenizer.json");
  try {
    const tokenizer = BPETokenizer.fromJSON(JSON.parse(await readFile(path, "utf8")));
    console.log(`Reusing tokenizer.json (vocab ${tokenizer.vocabSize})`);
    return tokenizer;
  } catch {
    // train below
  }
  console.log(`Collecting a ${(sampleBytes / 1024 / 1024).toFixed(0)} MB sample for BPE training...`);
  const pieces: string[] = [];
  let bytes = 0;
  for await (const text of docs(rawDir)) {
    pieces.push(text);
    bytes += text.length + 2;
    if (bytes >= sampleBytes) break;
  }
  const started = Date.now();
  const tokenizer = new BPETokenizer();
  tokenizer.train(pieces.join("\n\n"), vocabSize, (vocab, target) => {
    if (vocab % 64 === 0) process.stdout.write(`\r  vocab ${vocab}/${target}          `);
  });
  console.log(`\r  vocab ${tokenizer.vocabSize} in ${((Date.now() - started) / 1000).toFixed(1)}s`);
  await writeFile(path, JSON.stringify(tokenizer.toJSON()));
  return tokenizer;
}

/** Buffered Uint16 sink: batches token ids into large writes with backpressure. */
class TokenSink {
  private buf: number[] = [];
  count = 0;
  constructor(private stream: WriteStream, private flushAt = 4_000_000) {}
  async push(ids: number[]): Promise<void> {
    for (const id of ids) this.buf.push(id);
    this.count += ids.length;
    if (this.buf.length >= this.flushAt) await this.flush();
  }
  async flush(): Promise<void> {
    if (this.buf.length === 0) return;
    const chunk = Uint16Array.from(this.buf);
    this.buf = [];
    if (!this.stream.write(Buffer.from(chunk.buffer))) {
      await new Promise((r) => this.stream.once("drain", r));
    }
  }
  async close(): Promise<void> {
    await this.flush();
    await new Promise((resolve, reject) => this.stream.end(() => resolve(null)).once("error", reject));
  }
}

async function main(): Promise<void> {
  const targetBytes = parseNum("target-gb", 8) * 1024 * 1024 * 1024;
  const vocabSize = parseNum("vocab", 16384);
  const valEvery = parseNum("val-every", 20);
  const sampleBytes = parseNum("sample-mb", 48) * 1024 * 1024;
  const dir = resolve(join(import.meta.dirname, ".."), parseStr("dir", "data-big"));
  const rawDir = join(dir, "raw");
  await mkdir(rawDir, { recursive: true });

  console.log(`Target: ${(targetBytes / 1024 ** 3).toFixed(2)} GB of C4 text -> ${dir}\n`);

  const tokenizer = await ensureTokenizer(dir, rawDir, vocabSize, sampleBytes);
  if (tokenizer.vocabSize > 65536) throw new Error("Vocab exceeds Uint16 range");

  const train = new TokenSink(createWriteStream(join(dir, "train.bin")));
  const val = new TokenSink(createWriteStream(join(dir, "val.bin")));

  const started = Date.now();
  let bytes = 0;
  let nDocs = 0;
  for await (const text of docs(rawDir)) {
    // "\n\n" is the document separator, mirroring prepare-data.ts's join.
    const ids = tokenizer.encode(text + "\n\n");
    await (nDocs % valEvery === 0 ? val : train).push(ids);
    nDocs++;
    bytes += text.length + 2;
    if (nDocs % 2000 === 0) {
      const tokens = train.count + val.count;
      const rate = tokens / ((Date.now() - started) / 1000);
      process.stdout.write(
        `\r  ${nDocs} docs  ${(bytes / 1024 ** 3).toFixed(2)} GB  ${(tokens / 1e6).toFixed(1)}M tok  (${(rate / 1000).toFixed(0)}k tok/s)   `,
      );
    }
    if (bytes >= targetBytes) break;
  }
  await train.close();
  await val.close();

  const totalTokens = train.count + val.count;
  await writeFile(
    join(dir, "meta.json"),
    JSON.stringify(
      {
        source: "allenai/c4 en",
        vocabSize: tokenizer.vocabSize,
        docs: nDocs,
        corpusBytes: bytes,
        totalTokens,
        trainTokens: train.count,
        valTokens: val.count,
        bytesPerToken: bytes / totalTokens,
      },
      null,
      2,
    ),
  );

  console.log(`\n
Done in ${((Date.now() - started) / 60000).toFixed(1)} min.
  docs         ${nDocs}
  corpus       ${(bytes / 1024 ** 3).toFixed(2)} GB
  vocab        ${tokenizer.vocabSize}
  compression  ${(bytes / totalTokens).toFixed(2)} chars/token
  train        ${(train.count / 1e6).toFixed(2)}M tokens
  val          ${(val.count / 1e6).toFixed(2)}M tokens`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
