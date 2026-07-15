/**
 * Builds the training corpus: downloads public-domain books from Project
 * Gutenberg, strips their boilerplate, trains a BPE vocabulary, and writes the
 * tokenized corpus as a flat Uint16 array.
 *
 * Sizing rationale: the pure-CPU engine sustains ~2.3 GFLOP/s, and a GPT step
 * costs ~6 * params FLOPs per token, so a ~1M-parameter model consumes only a
 * few million tokens per hour. A ~60 MB corpus (~16M tokens) is therefore more
 * than a single training run can traverse — large enough that the model never
 * sees the same text twice, which is exactly what we want. Disk is not the
 * binding constraint here; compute is.
 *
 * Run: pnpm prepare-data [--target-mb 60] [--vocab 1024]
 *
 * Outputs (all under data/, which is gitignored):
 *   data/raw/<id>.txt   cached downloads, so re-runs cost nothing
 *   data/corpus.txt     cleaned, concatenated text
 *   data/tokenizer.json learned BPE merges
 *   data/train.bin      Uint16 token ids (90%)
 *   data/val.bin        Uint16 token ids (10%)
 *   data/meta.json      vocab size, token counts, book list
 */

import { mkdir, readFile, writeFile, stat } from "node:fs/promises";
import { join } from "node:path";
import { BPETokenizer } from "../src/tokenizer/bpe.js";

/** Public-domain Project Gutenberg ebook ids, spanning fiction, essays, and history. */
const BOOK_IDS = [
  1342, 84, 1661, 2701, 11, 1260, 98, 1400, 46, 730, 766, 786, 580, 963, 1023, 675, 967,
  74, 76, 3176, 102, 120, 158, 141, 105, 121, 161, 21839, 5200, 1080, 174, 345, 43, 209,
  215, 219, 244, 271, 289, 33, 36, 35, 5230, 164, 103, 18857, 2148, 932, 1064, 25344,
  2591, 503, 1727, 2199, 6130, 1497, 1656, 3207, 4300, 2600, 1399, 2554, 28054, 600,
  1184, 135, 2413, 1998, 4363, 8800, 6593, 3600, 3300, 2680, 1232, 2130, 10, 1250, 205,
  514, 20203, 3825, 829, 951, 1250, 30254, 16328, 2814, 4217, 2542, 1112, 1524, 1513,
  1533, 1531, 2264, 27827, 1041, 8492, 100, 3623, 5740, 2265, 12, 55, 113, 236, 1257,
  2852, 863, 61, 583, 696, 3268, 42, 15, 71, 160, 434, 1155, 2166, 4517, 7370, 8117,
];

const GUTENBERG_HEADER = /\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[\s\S]*?\*\*\*/i;
const GUTENBERG_FOOTER = /\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[\s\S]*$/i;

const DATA_DIR = join(import.meta.dirname, "..", "data");
const RAW_DIR = join(DATA_DIR, "raw");

function parseFlag(name: string, fallback: number): number {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? Number(process.argv[i + 1]) : fallback;
}

/** Downloads one book, caching it so re-runs never re-fetch. */
async function fetchBook(id: number): Promise<string | null> {
  const cachePath = join(RAW_DIR, `${id}.txt`);
  try {
    await stat(cachePath);
    return await readFile(cachePath, "utf8");
  } catch {
    // not cached yet
  }

  const urls = [
    `https://www.gutenberg.org/cache/epub/${id}/pg${id}.txt`,
    `https://www.gutenberg.org/files/${id}/${id}-0.txt`,
  ];
  for (const url of urls) {
    try {
      const response = await fetch(url, {
        signal: AbortSignal.timeout(45_000),
        headers: { "User-Agent": "turt-research/0.1 (educational LLM training)" },
      });
      if (!response.ok) continue;
      const text = await response.text();
      if (text.length < 10_000) continue; // too short to be a real book
      await writeFile(cachePath, text, "utf8");
      return text;
    } catch {
      continue;
    }
  }
  return null;
}

/** Removes Gutenberg's license header/footer and normalizes line endings. */
function clean(raw: string): string {
  let text = raw;
  const header = text.match(GUTENBERG_HEADER);
  if (header) text = text.slice(header.index! + header[0].length);
  text = text.replace(GUTENBERG_FOOTER, "");
  return text
    .replace(/\r\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n") // collapse long gaps
    .trim();
}

async function main(): Promise<void> {
  const targetBytes = parseFlag("target-mb", 60) * 1024 * 1024;
  const vocabSize = parseFlag("vocab", 1024);

  await mkdir(RAW_DIR, { recursive: true });

  console.log(`Target corpus: ${(targetBytes / 1024 / 1024).toFixed(0)} MB, vocab ${vocabSize}\n`);

  const pieces: string[] = [];
  const included: number[] = [];
  let bytes = 0;

  for (const id of BOOK_IDS) {
    if (bytes >= targetBytes) break;
    const raw = await fetchBook(id);
    if (!raw) {
      console.log(`  skip ${id} (unavailable)`);
      continue;
    }
    const text = clean(raw);
    if (text.length < 10_000) {
      console.log(`  skip ${id} (too short after cleaning)`);
      continue;
    }
    pieces.push(text);
    included.push(id);
    bytes += text.length;
    process.stdout.write(
      `\r  ${included.length} books, ${(bytes / 1024 / 1024).toFixed(1)} MB          `,
    );
  }

  const corpus = pieces.join("\n\n");
  console.log(`\n\nCorpus: ${included.length} books, ${(corpus.length / 1024 / 1024).toFixed(1)} MB`);
  await writeFile(join(DATA_DIR, "corpus.txt"), corpus, "utf8");

  // BPE merges converge quickly; training on a sample keeps this step to
  // seconds while producing the same vocabulary the full corpus would.
  const sample = corpus.slice(0, Math.min(corpus.length, 8 * 1024 * 1024));
  console.log(`\nTraining BPE on a ${(sample.length / 1024 / 1024).toFixed(1)} MB sample...`);
  const started = Date.now();
  const tokenizer = new BPETokenizer();
  tokenizer.train(sample, vocabSize, (vocab, target) => {
    if (vocab % 64 === 0) {
      process.stdout.write(`\r  vocab ${vocab}/${target}          `);
    }
  });
  console.log(`\r  vocab ${tokenizer.vocabSize} in ${((Date.now() - started) / 1000).toFixed(1)}s`);

  console.log("\nEncoding corpus...");
  const encodeStart = Date.now();
  // Encode in chunks aligned to line boundaries to bound peak memory.
  const CHUNK = 1024 * 1024;
  const ids: number[] = [];
  let offset = 0;
  while (offset < corpus.length) {
    let end = Math.min(offset + CHUNK, corpus.length);
    if (end < corpus.length) {
      const newline = corpus.lastIndexOf("\n", end);
      if (newline > offset) end = newline;
    }
    for (const id of tokenizer.encode(corpus.slice(offset, end))) ids.push(id);
    offset = end;
    process.stdout.write(
      `\r  ${((offset / corpus.length) * 100).toFixed(0)}%  ${(ids.length / 1e6).toFixed(2)}M tokens     `,
    );
  }
  const seconds = (Date.now() - encodeStart) / 1000;
  console.log(`\r  ${(ids.length / 1e6).toFixed(2)}M tokens in ${seconds.toFixed(1)}s          `);

  if (tokenizer.vocabSize > 65536) throw new Error("Vocab exceeds Uint16 range");
  const tokens = Uint16Array.from(ids);
  const split = Math.floor(tokens.length * 0.9);
  const train = tokens.subarray(0, split);
  const val = tokens.subarray(split);
  await writeFile(join(DATA_DIR, "train.bin"), Buffer.from(train.buffer, train.byteOffset, train.byteLength));
  await writeFile(join(DATA_DIR, "val.bin"), Buffer.from(val.buffer, val.byteOffset, val.byteLength));
  await writeFile(join(DATA_DIR, "tokenizer.json"), JSON.stringify(tokenizer.toJSON()));
  await writeFile(
    join(DATA_DIR, "meta.json"),
    JSON.stringify(
      {
        vocabSize: tokenizer.vocabSize,
        books: included.length,
        bookIds: included,
        corpusBytes: corpus.length,
        totalTokens: tokens.length,
        trainTokens: split,
        valTokens: tokens.length - split,
        bytesPerToken: corpus.length / tokens.length,
      },
      null,
      2,
    ),
  );

  console.log(`
Done.
  books        ${included.length}
  corpus       ${(corpus.length / 1024 / 1024).toFixed(1)} MB
  vocab        ${tokenizer.vocabSize}
  compression  ${(corpus.length / tokens.length).toFixed(2)} chars/token
  train        ${(split / 1e6).toFixed(2)}M tokens
  val          ${((tokens.length - split) / 1e6).toFixed(2)}M tokens`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
