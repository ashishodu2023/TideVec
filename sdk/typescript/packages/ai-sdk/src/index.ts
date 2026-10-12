import { embed as aiEmbed, embedMany as aiEmbedMany, tool, type LanguageModelMiddleware } from "ai";
import {
  AgentMemory,
  type AgentMemoryOptions,
  type Embedder,
  type EmbedFn,
  type Filter,
  type SearchHit,
  TideVec,
  type TideVecOptions,
  embedAll,
  toEmbedder,
} from "tidevec";
import { z } from "zod";

/** Embedding model as accepted by the installed AI SDK's `embed()` (model object or gateway id). */
export type AIEmbeddingModel = Parameters<typeof aiEmbed>[0]["model"];

/** An AI SDK embedding model, a TideVec {@link Embedder}, or a `(text) => vector` function. */
export type EmbeddingSource = AIEmbeddingModel | Embedder | EmbedFn;

function isTideVecEmbedder(e: unknown): e is Embedder {
  return typeof e === "object" && e !== null && typeof (e as Embedder).embed === "function";
}

/** Adapt any {@link EmbeddingSource} to a TideVec {@link Embedder}. */
export function asEmbedder(source: EmbeddingSource, dim?: number): Embedder {
  if (typeof source === "function") return toEmbedder(source as EmbedFn, dim);
  if (isTideVecEmbedder(source)) return source;
  const model = source as AIEmbeddingModel;
  return {
    dim,
    embed: async (value: string) => (await aiEmbed({ model, value })).embedding,
    embedMany: async (values: string[]) => (await aiEmbedMany({ model, values })).embeddings,
  };
}

function asClient(client: TideVec | TideVecOptions | string | undefined): TideVec {
  return client instanceof TideVec ? client : new TideVec(client ?? {});
}

// ---- memory tools ------------------------------------------------------

export interface MemoryToolsOptions extends Omit<AgentMemoryOptions, "embed"> {
  /** TideVec client, or options/URL to build one. Default `http://localhost:6399`. */
  client?: TideVec | TideVecOptions | string;
  embedding: EmbeddingSource;
  /** Reuse an existing AgentMemory instead of building one. */
  memory?: AgentMemory;
  /** Default number of memories `recall` returns. */
  defaultK?: number;
}

export const MEMORY_INSTRUCTIONS =
  "You have long-term memory. Call `recall` before answering questions that may depend on " +
  "earlier conversations. Call `remember` for durable facts about the user, project or " +
  "decisions; pass a stable `key` (e.g. 'user.timezone', 'project.db_choice') so that a later " +
  "update supersedes the old value instead of contradicting it.";

/**
 * `remember`, `recall`, `history` and `forget` tools backed by TideVec
 * {@link AgentMemory}. Facts stored under the same key supersede each
 * other, so recall returns the current value instead of every version.
 *
 * ```ts
 * const tools = tidevecMemoryTools({ embedding: openai.embedding("text-embedding-3-small") });
 * await generateText({ model, tools, system: MEMORY_INSTRUCTIONS, prompt });
 * ```
 */
export function tidevecMemoryTools(options: MemoryToolsOptions) {
  const { client, embedding, memory: given, defaultK = 5, ...memOpts } = options;
  const memory = given ?? new AgentMemory(asClient(client), { ...memOpts, embed: asEmbedder(embedding, memOpts.dim) });

  return {
    remember: tool({
      description:
        "Store a durable fact in long-term memory. Use `key` to name the fact (e.g. 'user.editor'); " +
        "a new memory with an existing key supersedes the previous value.",
      inputSchema: z.object({
        text: z.string().min(1).describe("The fact to remember, as a full sentence"),
        key: z.string().optional().describe("Stable name of the fact, e.g. 'user.timezone'"),
        metadata: z.record(z.string(), z.string()).optional().describe("Optional string tags"),
      }),
      execute: async ({ text, key, metadata }) => memory.remember(text, { key: key || undefined, metadata }),
    }),
    recall: tool({
      description:
        "Search long-term memory. Returns the current version of each relevant fact; outdated " +
        "versions are hidden unless includeSuperseded is true.",
      inputSchema: z.object({
        query: z.string().min(1).describe("What to look for"),
        k: z.number().int().min(1).max(50).optional().describe("Maximum memories to return"),
        key: z.string().optional().describe("Restrict to one fact key"),
        includeSuperseded: z.boolean().optional().describe("Also return outdated versions"),
      }),
      execute: async ({ query, k, key, includeSuperseded }) =>
        memory.recall(query, { k: k ?? defaultK, key: key || undefined, includeSuperseded }),
    }),
    history: tool({
      description: "Every stored version of the fact named `key`, oldest first.",
      inputSchema: z.object({ key: z.string().min(1) }),
      execute: async ({ key }) => memory.history(key),
    }),
    forget: tool({
      description: "Delete one memory by id, or every version of a fact key.",
      inputSchema: z.object({
        memoryId: z.string().optional(),
        key: z.string().optional(),
      }),
      execute: async ({ memoryId, key }) => {
        if (!memoryId && !key) return { deleted: 0, error: "pass memoryId or key" };
        return { deleted: await memory.forget({ id: memoryId || undefined, key: key || undefined }) };
      },
    }),
  };
}

// ---- retrieval ---------------------------------------------------------

export interface RetrievalOptions {
  client?: TideVec | TideVecOptions | string;
  collection: string;
  embedding: EmbeddingSource;
  /** Payload key holding the document text (default `"text"`). */
  textKey?: string;
  topK?: number;
  /** Drop documents a newer version supersedes (default true). */
  excludeSuperseded?: boolean;
  /** Recency weight per query; omit for the collection default. */
  temporalBlend?: number;
  /** Fixed payload filter applied to every query. */
  filter?: Filter;
}

export interface RetrievedDocument {
  id: string;
  text: string;
  score: number;
  createdAt: string;
  metadata: Record<string, string>;
  supersededBy: string[];
}

function toDocument(hit: SearchHit, textKey: string): RetrievedDocument {
  const { [textKey]: text = "", ...metadata } = hit.payload;
  return {
    id: hit.id,
    text,
    score: Math.round(hit.score * 10_000) / 10_000,
    createdAt: new Date(hit.createdAt).toISOString(),
    metadata,
    supersededBy: hit.supersededBy,
  };
}

/** Query a collection with text; the building block of the tool and middleware. */
export function createRetriever(options: RetrievalOptions) {
  const client = asClient(options.client);
  const embedder = asEmbedder(options.embedding);
  const textKey = options.textKey ?? "text";
  return async (query: string, o: { topK?: number; filter?: Filter; signal?: AbortSignal } = {}) => {
    const vector = await embedder.embed(query);
    const { hits } = await client.search(options.collection, vector, {
      topK: o.topK ?? options.topK ?? 5,
      excludeSuperseded: options.excludeSuperseded ?? true,
      temporalBlend: options.temporalBlend,
      filter: o.filter ?? options.filter,
      signal: o.signal,
    });
    return hits.map((h) => toDocument(h, textKey));
  };
}

/** A `searchKnowledge`-style tool over a TideVec collection. Outdated document versions are excluded. */
export function tidevecSearchTool(options: RetrievalOptions & { description?: string }) {
  const retrieve = createRetriever(options);
  return tool({
    description:
      options.description ??
      "Search the knowledge base. Returns the current version of each matching document with its date.",
    inputSchema: z.object({
      query: z.string().min(1).describe("Search query"),
      k: z.number().int().min(1).max(50).optional().describe("Number of documents"),
    }),
    execute: async ({ query, k }, { abortSignal }) => retrieve(query, { topK: k, signal: abortSignal }),
  });
}

export interface RagMiddlewareOptions extends RetrievalOptions {
  /** Build the system message from retrieved docs. */
  formatContext?: (docs: RetrievedDocument[], query: string) => string;
}

function defaultFormat(docs: RetrievedDocument[]): string {
  if (!docs.length) return "";
  const lines = docs.map((d, i) => `[${i + 1}] (${d.createdAt.slice(0, 10)}) ${d.text}`);
  return (
    "Use the following context when it is relevant. Each entry is the current version of a " +
    `document, with its date:\n${lines.join("\n")}`
  );
}

/**
 * Language-model middleware that retrieves context for the last user
 * message and prepends it as a system message.
 *
 * ```ts
 * const model = wrapLanguageModel({ model: openai("gpt-4o-mini"),
 *   middleware: tidevecRagMiddleware({ collection: "kb", embedding }) });
 * ```
 */
export function tidevecRagMiddleware(options: RagMiddlewareOptions): LanguageModelMiddleware {
  const retrieve = createRetriever(options);
  const format = options.formatContext ?? defaultFormat;
  return {
    transformParams: async ({ params }) => {
      const prompt = params.prompt;
      let query = "";
      for (let i = prompt.length - 1; i >= 0 && !query; i--) {
        const m = prompt[i]!;
        if (m.role === "user") {
          query = m.content
            .map((p) => (p.type === "text" ? p.text : ""))
            .join(" ")
            .trim();
        }
      }
      if (!query) return params;
      const context = format(await retrieve(query, { signal: params.abortSignal }), query);
      if (!context) return params;
      return { ...params, prompt: [{ role: "system" as const, content: context }, ...prompt] };
    },
  };
}

/** Embed and upsert documents so the retriever, tool and middleware can find them. */
export async function addDocuments(
  options: Pick<RetrievalOptions, "client" | "collection" | "embedding" | "textKey"> & {
    /** Create the collection (with supersession on `matchField`) if missing. */
    createIfMissing?: { dim: number; matchField?: string; similarityThreshold?: number };
  },
  docs: { id: string; text: string; metadata?: Record<string, unknown>; createdAt?: number | Date }[],
): Promise<number> {
  const client = asClient(options.client);
  const textKey = options.textKey ?? "text";
  if (options.createIfMissing) {
    const c = options.createIfMissing;
    await client.ensureCollection(options.collection, {
      dim: c.dim,
      supersession: c.matchField
        ? { matchField: c.matchField, similarityThreshold: c.similarityThreshold ?? 0.5 }
        : undefined,
    });
  }
  const vectors = await embedAll(asEmbedder(options.embedding), docs.map((d) => d.text));
  return client.upsert(
    options.collection,
    docs.map((d, i) => ({
      id: d.id,
      embedding: vectors[i]!,
      payload: { ...d.metadata, [textKey]: d.text },
      createdAt: d.createdAt,
    })),
  );
}

export { AgentMemory, TideVec };
