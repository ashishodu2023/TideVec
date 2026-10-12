/** Anything that turns text into a vector. */
export interface Embedder {
  /** Output dimension, if known without calling the model. */
  dim?: number;
  embed(text: string): Promise<number[]> | number[];
  embedMany?(texts: string[]): Promise<number[][]> | number[][];
}

export type EmbedFn = (text: string) => Promise<number[]> | number[];

export function toEmbedder(e: Embedder | EmbedFn, dim?: number): Embedder {
  return typeof e === "function" ? { dim, embed: e } : e;
}

export async function embedAll(e: Embedder, texts: string[]): Promise<number[][]> {
  if (e.embedMany) return e.embedMany(texts);
  return Promise.all(texts.map((t) => e.embed(t)));
}

const TOKEN = /[\p{L}\p{N}_]+/gu;
const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const MASK64 = 0xffffffffffffffffn;
const encoder = new TextEncoder();

function fnv1a64(s: string): bigint {
  let h = FNV_OFFSET;
  for (const b of encoder.encode(s)) {
    h ^= BigInt(b);
    h = (h * FNV_PRIME) & MASK64;
  }
  return h;
}

/**
 * Deterministic feature-hashing embedder over word unigrams and bigrams.
 * No model and no network: near-identical texts get near-identical
 * vectors, but there is no notion of synonyms. For tests and demos only.
 * (Same idea as Python's `HashingEmbedder`, but the vectors differ.)
 */
export class HashingEmbedder implements Embedder {
  constructor(readonly dim = 384) {
    if (dim <= 0) throw new Error("dim must be positive");
  }

  embed(text: string): number[] {
    const words = text.toLowerCase().match(TOKEN) ?? [];
    const feats = [...words];
    for (let i = 0; i + 1 < words.length; i++) feats.push(`${words[i]} ${words[i + 1]}`);
    const vec = new Array<number>(this.dim).fill(0);
    for (const f of feats) {
      const h = fnv1a64(f);
      const idx = Number(h % BigInt(this.dim));
      vec[idx]! += (h >> 63n) & 1n ? 1 : -1;
    }
    const norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0));
    return norm > 0 ? vec.map((x) => x / norm) : vec;
  }

  embedMany(texts: string[]): number[][] {
    return texts.map((t) => this.embed(t));
  }
}
