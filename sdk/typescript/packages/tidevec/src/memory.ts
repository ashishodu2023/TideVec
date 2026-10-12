import type { TideVec } from "./client.js";
import { payloadStr } from "./client.js";
import { type EmbedFn, type Embedder, toEmbedder } from "./embeddings.js";
import { HalfLife, type SearchHit } from "./types.js";

// Payload layout shared with the Python AgentMemory and `tidevec-mcp`,
// so either side can read memories the other wrote.
const KEY = "memory_key";
const TEXT = "text";

export interface AgentMemoryOptions {
  /** Embedding model: an {@link Embedder} or a plain `(text) => vector` function. */
  embed: Embedder | EmbedFn;
  collection?: string;
  /** Embedding dimension; probed with one embed call when omitted. */
  dim?: number;
  halfLifeMs?: number;
  /** Weight of recency in recall ranking. */
  temporalBlend?: number;
  /** vector_score at which an unkeyed memory counts as a restatement (0.95 = cos 0.9). */
  supersedeThreshold?: number;
}

export interface RememberOptions {
  /** Names the fact; a new memory with the same key supersedes earlier ones. */
  key?: string;
  metadata?: Record<string, unknown>;
  createdAt?: number | Date;
}

export interface RecallOptions {
  k?: number;
  includeSuperseded?: boolean;
  key?: string;
}

export interface Memory {
  id: string;
  text: string;
  key: string | null;
  score: number;
  /** ISO-8601 timestamp. */
  createdAt: string;
  age: string;
  stale: boolean;
  supersededBy: string[];
  contradictedBy: string[];
  metadata: Record<string, string>;
}

function randomId(): string {
  const c = (globalThis as { crypto?: { randomUUID?: () => string } }).crypto;
  if (c?.randomUUID) return c.randomUUID().replace(/-/g, "");
  let s = "";
  for (let i = 0; i < 32; i++) s += Math.floor(Math.random() * 16).toString(16);
  return s;
}

function age(createdAt: number): string {
  const secs = Math.max(0, Math.floor((Date.now() - createdAt) / 1000));
  for (const [unit, size] of [["d", 86400], ["h", 3600], ["m", 60]] as const) {
    if (secs >= size) return `${Math.floor(secs / size)}${unit} ago`;
  }
  return "just now";
}

/**
 * Versioned long-term memory for agents, on top of a TideVec server.
 *
 * ```ts
 * const mem = new AgentMemory(new TideVec(), { embed: myEmbedder });
 * await mem.remember("User's favourite editor is Vim", { key: "user.editor" });
 * await mem.remember("User switched to Helix", { key: "user.editor" });
 * await mem.recall("which editor does the user like?"); // -> Helix only
 * ```
 */
export class AgentMemory {
  readonly collection: string;
  private readonly embedder: Embedder;
  private ready?: Promise<void>;

  constructor(
    readonly client: TideVec,
    private readonly options: AgentMemoryOptions,
  ) {
    this.collection = options.collection ?? "agent_memory";
    this.embedder = toEmbedder(options.embed, options.dim);
  }

  private embed(text: string): Promise<number[]> {
    return Promise.resolve(this.embedder.embed(text));
  }

  /** Create the collection if needed. Called lazily by every method. */
  init(): Promise<void> {
    this.ready ??= (async () => {
      if (await this.client.hasCollection(this.collection)) return;
      const dim = this.options.dim ?? this.embedder.dim ?? (await this.embed("probe")).length;
      await this.client.ensureCollection(this.collection, {
        dim,
        temporal: {
          halfLifeMs: this.options.halfLifeMs ?? HalfLife.ONE_MONTH,
          temporalBlend: this.options.temporalBlend ?? 0.2,
        },
        supersession: { similarityThreshold: this.options.supersedeThreshold ?? 0.95 },
      });
    })().catch((e) => {
      this.ready = undefined;
      throw e;
    });
    return this.ready;
  }

  private async versions(key: string, emb: number[]): Promise<SearchHit[]> {
    const { hits } = await this.client.search(this.collection, emb, {
      topK: 200,
      filter: `${KEY}=${key}`,
      temporalBlend: 0,
      includeStalenessWarnings: false,
    });
    return hits;
  }

  private static toMemory(hit: SearchHit): Memory {
    const { [TEXT]: text = "", [KEY]: key = "", ...metadata } = hit.payload;
    return {
      id: hit.id,
      text,
      key: key || null,
      score: Math.round(hit.score * 10_000) / 10_000,
      createdAt: new Date(hit.createdAt).toISOString(),
      age: age(hit.createdAt),
      stale: hit.stalenessWarning,
      supersededBy: hit.supersededBy,
      contradictedBy: hit.contradictedBy,
      metadata,
    };
  }

  /** Store a memory. Returns its id and the ids it superseded. */
  async remember(text: string, o: RememberOptions = {}): Promise<{ id: string; superseded: string[] }> {
    if (!text.trim()) throw new Error("text must not be empty");
    await this.init();
    const id = randomId();
    const ts = o.createdAt instanceof Date ? o.createdAt.getTime() : (o.createdAt ?? Date.now());
    const emb = await this.embed(text);
    const payload: Record<string, string> = {};
    for (const [k, v] of Object.entries(o.metadata ?? {})) payload[k] = payloadStr(v);
    payload[TEXT] = text;
    payload[KEY] = o.key ?? "";

    const superseded: string[] = [];
    const edges: { src: string; tgt: string; type: "UPDATES"; weight: number }[] = [];
    if (o.key) {
      for (const prior of await this.versions(o.key, emb)) {
        if (prior.createdAt <= ts) {
          if (!prior.isSuperseded) {
            superseded.push(prior.id);
            edges.push({ src: id, tgt: prior.id, type: "UPDATES", weight: 1 });
          }
        } else {
          edges.push({ src: prior.id, tgt: id, type: "UPDATES", weight: 1 });
        }
      }
    }
    await this.client.upsert(this.collection, [{ id, embedding: emb, payload, createdAt: ts }]);
    if (edges.length) await this.client.addEdges(this.collection, edges);
    return { id, superseded };
  }

  /** Most relevant memories for `query`, newest versions only by default. */
  async recall(query: string, o: RecallOptions = {}): Promise<Memory[]> {
    await this.init();
    const k = o.k ?? 5;
    const emb = await this.embed(query);
    const { hits } = await this.client.search(this.collection, emb, {
      topK: o.includeSuperseded ? k : Math.max(k * 4, 20),
      filter: o.key ? `${KEY}=${o.key}` : undefined,
    });
    const kept = o.includeSuperseded ? hits : hits.filter((h) => !h.isSuperseded);
    return kept.slice(0, k).map(AgentMemory.toMemory);
  }

  /** Every version of a key, oldest first. */
  async history(key: string): Promise<Memory[]> {
    await this.init();
    const hits = await this.versions(key, await this.embed(key));
    return hits.sort((a, b) => a.createdAt - b.createdAt).map(AgentMemory.toMemory);
  }

  /** Delete one memory by id, or every version of a key. Returns the number removed. */
  async forget(o: { id?: string; key?: string }): Promise<number> {
    if (!o.id && !o.key) throw new Error("forget() needs id or key");
    await this.init();
    let n = 0;
    if (o.id) n += await this.client.delete(this.collection, { ids: [o.id] });
    if (o.key) n += await this.client.delete(this.collection, { filter: `${KEY}=${o.key}` });
    return n;
  }
}
