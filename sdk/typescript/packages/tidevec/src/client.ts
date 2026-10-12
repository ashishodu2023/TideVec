import {
  NotLeaderError,
  RateLimitError,
  TideVecConnectionError,
  TideVecError,
  errorFromResponse,
} from "./errors.js";
import type {
  Alias,
  AsOf,
  BackupManifest,
  CollectionInfo,
  CreateCollectionOptions,
  DriftStartOptions,
  DriftStatus,
  Edge,
  EdgeType,
  Filter,
  GetOptions,
  HealthResponse,
  Payload,
  PayloadInput,
  ScrollOptions,
  ScrollPage,
  SearchHit,
  SearchOptions,
  SearchResponse,
  StoredVector,
  TemporalOptions,
  VectorInput,
} from "./types.js";

export type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

export interface TideVecOptions {
  /** `"http://host:port"`, `"host:port"` or `"host"` (port 6399). Default `http://localhost:6399`. */
  url?: string;
  /** Sent as `X-Api-Key`. Defaults to `$TIDEVEC_API_KEY` on Node. */
  apiKey?: string;
  /** Per-request timeout in milliseconds (default 30000). */
  timeoutMs?: number;
  /** Retries for HTTP 429 responses (default 3). */
  maxRetries?: number;
  /** Custom fetch implementation (default `globalThis.fetch`). */
  fetch?: FetchLike;
  /** Extra headers sent with every request. */
  headers?: Record<string, string>;
  /** Cluster mode: how long a write keeps following leader hints during an election (default 10000). */
  leaderWaitMs?: number;
}

export interface RequestOptions {
  signal?: AbortSignal;
  /** Override the client timeout for this request. */
  timeoutMs?: number;
}

export interface DeleteOptions {
  ids?: string[];
  filter?: Filter;
}

export interface CountOptions extends RequestOptions {
  filter?: Filter;
  asOf?: AsOf;
}

const DEFAULT_PORT = 6399;
const MAX_FETCH_FACTOR = 4;

/** How a payload value is stored: strings as-is, everything else JSON-encoded. */
export function payloadStr(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}

/** Normalise a base URL the same way the Python client does. */
export function normaliseUrl(url: string): string {
  let u = url.trim().replace(/\/+$/, "");
  if (!/^https?:\/\//i.test(u)) {
    const hostPart = u.split("/")[0] ?? u;
    // host or host:port without a scheme; IPv6 literals use [::1]:port
    const hasPort = /:\d+$/.test(hostPart);
    u = `http://${hasPort ? u : u.replace(hostPart, `${hostPart}:${DEFAULT_PORT}`)}`;
  }
  return u;
}

/**
 * Encode a filter for the request body. Single-key equality objects
 * become the `key=value` form every server understands; strings and
 * other objects pass through.
 */
export function encodeFilter(filter: Filter | undefined): string | Record<string, unknown> | undefined {
  if (filter === undefined || filter === null || filter === "") return undefined;
  if (typeof filter === "string") {
    if (!filter.includes("=") || filter.startsWith("=")) {
      throw new TideVecError(`filter must look like "key=value", got ${JSON.stringify(filter)}`);
    }
    return filter;
  }
  const keys = Object.keys(filter);
  if (keys.length === 1) {
    const key = keys[0]!;
    const value = filter[key];
    const scalar =
      typeof value === "string" ||
      typeof value === "number" ||
      typeof value === "boolean";
    if (scalar && !key.startsWith("$")) return `${key}=${payloadStr(value)}`;
  }
  return filter;
}

function ms(t: number | Date | undefined): number | undefined {
  if (t === undefined) return undefined;
  return t instanceof Date ? t.getTime() : Math.trunc(t);
}

function asOfToWire(t: AsOf | undefined): number | string | undefined {
  if (t === undefined) return undefined;
  return t instanceof Date ? t.getTime() : t;
}

function filterFields(o: { filter?: Filter; asOf?: AsOf }): Record<string, unknown> {
  return stripUndefined({ filter: encodeFilter(o.filter), as_of: asOfToWire(o.asOf) });
}

function stripUndefined<T extends Record<string, unknown>>(obj: T): T {
  for (const k of Object.keys(obj)) if (obj[k] === undefined) delete obj[k];
  return obj;
}

function encodePayload(payload: PayloadInput | undefined): Payload | undefined {
  if (!payload) return undefined;
  const out: Payload = {};
  for (const [k, v] of Object.entries(payload)) if (v !== undefined) out[k] = payloadStr(v);
  return out;
}

export function vectorToWire(v: VectorInput): Record<string, unknown> {
  return stripUndefined({
    id: v.id,
    embedding: Array.from(v.embedding, Number),
    payload: encodePayload(v.payload),
    created_at: ms(v.createdAt),
    valid_from: ms(v.validFrom),
    valid_until: ms(v.validUntil),
    ttl_seconds: v.ttlSeconds,
    edges: v.edges?.map((e) => ({ target_id: e.targetId, type: e.type, weight: e.weight ?? 1 })),
  });
}

function temporalToWire(t: TemporalOptions): Record<string, unknown> {
  return stripUndefined({
    half_life_ms: t.halfLifeMs,
    temporal_blend: t.temporalBlend,
    staleness_threshold: t.stalenessThreshold,
    temporal_cutoff: t.temporalCutoff,
  });
}

export function collectionToWire(name: string, o: CreateCollectionOptions): Record<string, unknown> {
  const body: Record<string, unknown> = stripUndefined({
    name,
    dim: o.dim,
    index_type: o.indexType,
    metric: o.metric,
    n_shards: o.nShards,
    n_replicas: o.nReplicas,
    write_quorum: o.writeQuorum,
    text_field: o.textField,
    payload_index: o.payloadIndex,
  });
  if (o.temporal) body.temporal = temporalToWire(o.temporal);
  if (o.supersession) {
    const s = o.supersession === true ? {} : o.supersession;
    body.supersession = stripUndefined({
      enabled: s.enabled ?? true,
      similarity_threshold: s.similarityThreshold,
      match_field: s.matchField,
      candidates: s.candidates,
    });
  }
  if (o.agent) {
    body.agent = stripUndefined({
      enabled: o.agent.enabled ?? true,
      agent_id: o.agent.agentId,
      memory_ttl_seconds: o.agent.memoryTtlSeconds,
      dedup_threshold: o.agent.dedupThreshold,
      cross_agent_readable: o.agent.crossAgentReadable,
    });
  }
  return body;
}

export function searchToWire(vector: ArrayLike<number> | null, o: SearchOptions = {}): Record<string, unknown> {
  const topK = o.topK ?? 10;
  if (vector === null && !o.text) throw new TideVecError("search needs a vector and/or options.text");
  const h = o.hybrid;
  return stripUndefined({
    vector: vector === null ? undefined : Array.from(vector, Number),
    text: o.text,
    hybrid: h ? stripUndefined({ fusion: h.fusion, alpha: h.alpha, rrf_k: h.rrfK, prefetch: h.prefetch }) : undefined,
    as_of: asOfToWire(o.asOf),
    top_k: o.excludeSuperseded ? Math.max(topK * MAX_FETCH_FACTOR, 20) : topK,
    temporal_blend: o.temporalBlend,
    superseded_penalty: o.supersededPenalty,
    mode: o.mode,
    causal_hops: o.causalHops,
    filter: encodeFilter(o.filter),
    metric: o.metric,
    ef_search: o.efSearch,
    include_trace: o.includeTrace,
    include_staleness_warnings: o.includeStalenessWarnings,
    ...o.extra,
  });
}

type Json = Record<string, unknown>;

export function hitFromWire(j: Json): SearchHit {
  const supersededBy = (j.superseded_by as string[] | undefined) ?? [];
  return {
    id: String(j.id),
    score: Number(j.score ?? 0),
    vectorScore: Number(j.vector_score ?? 0),
    temporalScore: Number(j.temporal_score ?? 0),
    createdAt: Number(j.created_at ?? 0),
    payload: (j.payload as Payload | undefined) ?? {},
    stalenessWarning: Boolean(j.staleness_warning),
    stalenessReason: String(j.staleness_reason ?? ""),
    causalNeighbors: (j.causal_neighbors as string[] | undefined) ?? [],
    contradictedBy: (j.contradicted_by as string[] | undefined) ?? [],
    supersededBy,
    isSuperseded: supersededBy.length > 0,
  };
}

export function storedFromWire(j: Json): StoredVector {
  const edges = ((j.edges as Json[] | undefined) ?? []).map((e) =>
    stripUndefined({
      targetId: String(e.target_id),
      type: e.type as EdgeType,
      weight: Number(e.weight ?? 1),
      createdAt: e.created_at as number | undefined,
    }),
  );
  const supersededBy = (j.superseded_by as string[] | undefined) ?? [];
  return stripUndefined({
    id: String(j.id),
    createdAt: Number(j.created_at ?? 0),
    validFrom: Number(j.valid_from ?? j.created_at ?? 0),
    validUntil: j.valid_until as number | undefined,
    payload: (j.payload as Payload | undefined) ?? {},
    embedding: j.embedding as number[] | undefined,
    edges,
    supersededBy,
    supersedes: (j.supersedes as string[] | undefined) ?? [],
    isSuperseded: supersededBy.length > 0,
  });
}

function collectionFromWire(c: Json): CollectionInfo {
  return stripUndefined({
    name: String(c.name),
    nVectors: Number(c.n_vectors ?? 0),
    nShards: Number(c.n_shards ?? 0),
    dim: c.dim as number | undefined,
    indexType: c.index_type as string | undefined,
    metric: c.metric as string | undefined,
    backend: c.backend as string | undefined,
    totalWrites: c.total_writes as number | undefined,
    totalQueries: c.total_queries as number | undefined,
    supersession: c.supersession as Record<string, unknown> | undefined,
    textField: (c.text_field as string | undefined) || undefined,
    aliases: c.aliases as string[] | undefined,
  });
}

const sleep = (msec: number) => new Promise((r) => setTimeout(r, msec));

function envApiKey(): string {
  const proc = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process;
  return proc?.env?.TIDEVEC_API_KEY ?? "";
}

/**
 * TideVec REST client.
 *
 * ```ts
 * const db = new TideVec({ url: "localhost:6399", apiKey: "..." });
 * await db.createCollection("docs", { dim: 384, supersession: { matchField: "doc_id" } });
 * await db.upsert("docs", [{ id: "v1", embedding, payload: { doc_id: "refunds" } }]);
 * const { hits } = await db.search("docs", query, { topK: 5, excludeSuperseded: true });
 * ```
 */
export class TideVec {
  readonly baseUrl: string;
  private readonly apiKey: string;
  private readonly timeoutMs: number;
  private readonly maxRetries: number;
  private readonly fetchImpl: FetchLike;
  private readonly headers: Record<string, string>;
  private readonly leaderWaitMs: number;

  constructor(options: TideVecOptions | string = {}) {
    const o: TideVecOptions = typeof options === "string" ? { url: options } : options;
    this.baseUrl = normaliseUrl(o.url ?? `http://localhost:${DEFAULT_PORT}`);
    this.apiKey = o.apiKey ?? envApiKey();
    this.timeoutMs = o.timeoutMs ?? 30_000;
    this.maxRetries = o.maxRetries ?? 3;
    const f = o.fetch ?? (globalThis.fetch as FetchLike | undefined);
    if (!f) throw new TideVecError("No fetch implementation: use Node 18+ or pass options.fetch");
    this.fetchImpl = f;
    this.headers = o.headers ?? {};
    this.leaderWaitMs = o.leaderWaitMs ?? 10_000;
  }

  // ---- transport ------------------------------------------------------

  /**
   * Send a request and return the parsed JSON body (the full envelope,
   * `{status, data}` for /v1 routes). Public so endpoints newer than this
   * client can be called directly.
   */
  async request<T = unknown>(
    method: string,
    path: string,
    body?: unknown,
    opts: RequestOptions = {},
  ): Promise<T> {
    let base = this.baseUrl;
    let attempt = 0;
    let hops = 0;
    const redirectDeadline = Date.now() + this.leaderWaitMs;
    for (;;) {
      try {
        return await this.requestOnce<T>(method, base + path, body, opts);
      } catch (e) {
        if (e instanceof RateLimitError && attempt < this.maxRetries) {
          await sleep(Math.min(2 ** attempt, 8) * 1000);
          attempt++;
          continue;
        }
        if (e instanceof NotLeaderError && Date.now() < redirectDeadline) {
          hops++;
          const hint = e.leader ? e.leader.replace(/\/+$/, "") : null;
          if (hint && hint !== base && hops <= 2) {
            base = hint;
            continue;
          }
          await sleep(Math.min(50 * hops, 500));
          base = hint ?? this.baseUrl;
          continue;
        }
        if (e instanceof TideVecConnectionError && base !== this.baseUrl && Date.now() < redirectDeadline) {
          await sleep(100);
          base = this.baseUrl;
          continue;
        }
        throw e;
      }
    }
  }

  private async requestOnce<T>(method: string, url: string, body: unknown, opts: RequestOptions): Promise<T> {
    const controller = new AbortController();
    const timeout = opts.timeoutMs ?? this.timeoutMs;
    const timer = setTimeout(() => controller.abort(new Error(`timed out after ${timeout} ms`)), timeout);
    const onAbort = () => controller.abort(opts.signal?.reason);
    if (opts.signal) {
      if (opts.signal.aborted) onAbort();
      else opts.signal.addEventListener("abort", onAbort, { once: true });
    }
    const headers: Record<string, string> = { Accept: "application/json", ...this.headers };
    if (body !== undefined) headers["Content-Type"] = "application/json";
    if (this.apiKey) headers["X-Api-Key"] = this.apiKey;

    let res: Response;
    let text: string;
    try {
      res = await this.fetchImpl(url, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      });
      text = await res.text();
    } catch (e) {
      if (opts.signal?.aborted) throw e;
      const reason = e instanceof Error ? (e.cause instanceof Error ? e.cause.message : e.message) : String(e);
      throw new TideVecConnectionError(`Cannot connect to TideVec at ${url}: ${reason}`, { cause: e });
    } finally {
      clearTimeout(timer);
      opts.signal?.removeEventListener("abort", onAbort);
    }

    let parsed: unknown = text;
    if (text) {
      try {
        parsed = JSON.parse(text);
      } catch {
        // non-JSON body (e.g. /metrics or a proxy error page)
      }
    }
    if (!res.ok) throw errorFromResponse(res.status, parsed);
    return (text ? parsed : {}) as T;
  }

  private async data<T>(method: string, path: string, body?: unknown, opts?: RequestOptions): Promise<T> {
    const r = await this.request<{ data?: T }>(method, path, body, opts);
    return (r && typeof r === "object" && "data" in r ? r.data : r) as T;
  }

  private static seg(name: string): string {
    return encodeURIComponent(name);
  }

  private col(name: string, suffix = ""): string {
    return `/v1/collections/${TideVec.seg(name)}${suffix}`;
  }

  // ---- health / info --------------------------------------------------

  health(opts?: RequestOptions): Promise<HealthResponse> {
    return this.request<HealthResponse>("GET", "/health", undefined, opts);
  }

  /** True if the server answers /health. */
  async ping(opts?: RequestOptions): Promise<boolean> {
    try {
      await this.health(opts);
      return true;
    } catch (e) {
      if (e instanceof TideVecError) return false;
      throw e;
    }
  }

  info(opts?: RequestOptions): Promise<Record<string, unknown>> {
    return this.data("GET", "/v1/info", undefined, opts);
  }

  /** Prometheus metrics text. */
  metrics(opts?: RequestOptions): Promise<string> {
    return this.request<string>("GET", "/metrics", undefined, opts);
  }

  /** Raft status (experimental cluster mode); `{mode: "standalone"}` on a single server. */
  clusterStatus(opts?: RequestOptions): Promise<Record<string, unknown>> {
    return this.data("GET", "/v1/cluster/status", undefined, opts);
  }

  // ---- collections ----------------------------------------------------

  async createCollection(name: string, options: CreateCollectionOptions, opts?: RequestOptions): Promise<string> {
    const d = await this.data<{ name: string }>("POST", "/v1/collections", collectionToWire(name, options), opts);
    return d.name;
  }

  /** Create the collection unless it exists. Returns true if it was created. */
  async ensureCollection(name: string, options: CreateCollectionOptions, opts?: RequestOptions): Promise<boolean> {
    if (await this.hasCollection(name, opts)) return false;
    try {
      await this.createCollection(name, options, opts);
      return true;
    } catch (e) {
      if (await this.hasCollection(name, opts)) return false; // lost a creation race
      throw e;
    }
  }

  async listCollections(opts?: RequestOptions): Promise<CollectionInfo[]> {
    const d = await this.data<Json[]>("GET", "/v1/collections", undefined, opts);
    return (d ?? []).map(collectionFromWire);
  }

  async getCollection(name: string, opts?: RequestOptions): Promise<CollectionInfo> {
    return collectionFromWire(await this.data<Json>("GET", this.col(name), undefined, opts));
  }

  async hasCollection(name: string, opts?: RequestOptions): Promise<boolean> {
    try {
      await this.request("GET", this.col(name), undefined, opts);
      return true;
    } catch (e) {
      if (e instanceof TideVecError && "status" in e && e.status === 404) return false;
      throw e;
    }
  }

  /** Drop a collection. Returns false if it did not exist. */
  async dropCollection(name: string, opts?: RequestOptions): Promise<boolean> {
    try {
      await this.request("DELETE", this.col(name), undefined, opts);
      return true;
    } catch (e) {
      if (e instanceof TideVecError && "status" in e && e.status === 404) return false;
      throw e;
    }
  }

  async setTemporal(name: string, temporal: TemporalOptions, opts?: RequestOptions): Promise<void> {
    await this.request("PUT", this.col(name, "/temporal"), temporalToWire(temporal), opts);
  }

  /**
   * Number of vectors. With `filter` or `asOf` this uses `POST .../count`
   * (servers with structured filters); otherwise the collection stats.
   */
  async count(collection: string, opts: CountOptions = {}): Promise<number> {
    if (opts.filter === undefined && opts.asOf === undefined) {
      return (await this.getCollection(collection, opts)).nVectors;
    }
    const d = await this.data<{ count: number }>("POST", this.col(collection, "/count"), filterFields(opts), opts);
    return d.count;
  }

  // ---- aliases --------------------------------------------------------

  async listAliases(opts?: RequestOptions): Promise<Alias[]> {
    return (await this.data<{ aliases?: Alias[] }>("GET", "/v1/aliases", undefined, opts)).aliases ?? [];
  }

  /** Create an alias or atomically repoint it to another collection. */
  async setAlias(alias: string, collection: string, opts?: RequestOptions): Promise<void> {
    await this.request("PUT", `/v1/aliases/${TideVec.seg(alias)}`, { collection }, opts);
  }

  async deleteAlias(alias: string, opts?: RequestOptions): Promise<boolean> {
    try {
      await this.request("DELETE", `/v1/aliases/${TideVec.seg(alias)}`, undefined, opts);
      return true;
    } catch (e) {
      if (e instanceof TideVecError && "status" in e && e.status === 404) return false;
      throw e;
    }
  }

  // ---- vectors --------------------------------------------------------

  /** Upsert vectors; returns the number written. Large arrays are sent in batches. */
  async upsert(
    collection: string,
    vectors: VectorInput[],
    opts: RequestOptions & { batchSize?: number } = {},
  ): Promise<number> {
    const batch = opts.batchSize ?? 500;
    let n = 0;
    for (let i = 0; i < vectors.length; i += batch) {
      const d = await this.data<{ inserted: number }>(
        "POST",
        this.col(collection, "/upsert"),
        { vectors: vectors.slice(i, i + batch).map(vectorToWire) },
        opts,
      );
      n += d.inserted;
    }
    return n;
  }

  /** Delete by ids and/or payload filter; returns the number removed. */
  async delete(collection: string, what: DeleteOptions | string[], opts?: RequestOptions): Promise<number> {
    const o: DeleteOptions = Array.isArray(what) ? { ids: what } : what;
    if (o.ids === undefined && o.filter === undefined) {
      throw new TideVecError("delete() needs ids or filter");
    }
    const body = stripUndefined({ ids: o.ids, filter: encodeFilter(o.filter) });
    const d = await this.data<{ deleted: number }>("POST", this.col(collection, "/delete"), body, opts);
    return d.deleted;
  }

  /** One stored vector, or null if the id does not exist. */
  async get(collection: string, id: string, o: GetOptions = {}): Promise<StoredVector | null> {
    // POST /get rather than GET /vectors/{id}: a missing id is reported in
    // `missing` instead of a 404 that looks like a missing collection.
    const { vectors } = await this.getMany(collection, [id], { withVector: true, ...o });
    return vectors[0] ?? null;
  }

  /** Fetch vectors by id, in request order; unknown ids are listed in `missing`. */
  async getMany(
    collection: string,
    ids: string[],
    o: GetOptions = {},
  ): Promise<{ vectors: StoredVector[]; missing: string[] }> {
    const body = stripUndefined({ ids, with_vector: o.withVector ?? false, with_payload: o.withPayload });
    const d = await this.data<{ vectors?: Json[]; missing?: string[] }>("POST", this.col(collection, "/get"), body, o);
    return { vectors: (d.vectors ?? []).map(storedFromWire), missing: d.missing ?? [] };
  }

  /** One page of vectors in id order. Pass `nextCursor` back as `cursor` until it is null. */
  async scrollPage(collection: string, o: ScrollOptions = {}): Promise<ScrollPage<StoredVector>> {
    const body = stripUndefined({
      ...filterFields(o),
      limit: o.limit,
      cursor: o.cursor ?? undefined,
      with_vector: o.withVector,
      with_payload: o.withPayload,
    });
    const d = await this.data<{ vectors?: Json[]; next_cursor?: string | null }>(
      "POST",
      this.col(collection, "/scroll"),
      body,
      o,
    );
    return { items: (d.vectors ?? []).map(storedFromWire), nextCursor: d.next_cursor ?? null };
  }

  /** Iterate over every matching vector: `for await (const v of db.scroll("kb", { filter })) ...` */
  async *scroll(collection: string, o: ScrollOptions = {}): AsyncGenerator<StoredVector> {
    let cursor = o.cursor ?? null;
    do {
      const page = await this.scrollPage(collection, { ...o, cursor });
      yield* page.items;
      cursor = page.nextCursor;
    } while (cursor);
  }

  /** One page of ids in id order. */
  async listIds(
    collection: string,
    o: Omit<ScrollOptions, "withVector" | "withPayload"> = {},
  ): Promise<ScrollPage<string>> {
    const body = stripUndefined({ ...filterFields(o), limit: o.limit, cursor: o.cursor ?? undefined });
    const d = await this.data<{ ids?: string[]; next_cursor?: string | null }>(
      "POST",
      this.col(collection, "/ids"),
      body,
      o,
    );
    return { items: d.ids ?? [], nextCursor: d.next_cursor ?? null };
  }

  /** Iterate over every matching id. */
  async *ids(collection: string, o: Omit<ScrollOptions, "withVector" | "withPayload"> = {}): AsyncGenerator<string> {
    let cursor = o.cursor ?? null;
    do {
      const page = await this.listIds(collection, { ...o, cursor });
      yield* page.items;
      cursor = page.nextCursor;
    } while (cursor);
  }

  async addEdges(collection: string, edges: Edge[], opts?: RequestOptions): Promise<number> {
    const body = { edges: edges.map((e) => ({ src: e.src, tgt: e.tgt, type: e.type, weight: e.weight ?? 1 })) };
    const d = await this.data<{ added: number }>("POST", this.col(collection, "/edges"), body, opts);
    return d.added;
  }

  // ---- search ---------------------------------------------------------

  /**
   * Dense, keyword (`options.text` with a `null` vector) or hybrid
   * (vector plus `options.text`) search.
   */
  async search(
    collection: string,
    vector: ArrayLike<number> | null,
    options: SearchOptions = {},
  ): Promise<SearchResponse> {
    const d = await this.data<{ results?: Json[]; trace?: Json }>(
      "POST",
      this.col(collection, "/search"),
      searchToWire(vector, options),
      { signal: options.signal },
    );
    let hits = (d.results ?? []).map(hitFromWire);
    if (options.excludeSuperseded) hits = hits.filter((h) => !h.isSuperseded).slice(0, options.topK ?? 10);
    return d.trace ? { hits, trace: d.trace } : { hits };
  }

  // ---- DriftBridge ----------------------------------------------------

  /** Start a zero-downtime re-embedding migration (standalone servers only). */
  startDrift(collection: string, o: DriftStartOptions, opts?: RequestOptions): Promise<Record<string, unknown>> {
    const body = stripUndefined({
      reembed_url: o.reembedUrl,
      M: o.M,
      ef_construction: o.efConstruction,
      new_dim: o.newDim,
      temporal: o.temporal ? temporalToWire(o.temporal) : undefined,
    });
    return this.data("POST", this.col(collection, "/drift/start"), body, opts);
  }

  async driftStatus(collection: string, opts?: RequestOptions): Promise<DriftStatus> {
    const d = await this.data<Json>("GET", this.col(collection, "/drift/status"), undefined, opts);
    return {
      collection: String(d.collection ?? collection),
      phase: (d.phase as DriftStatus["phase"]) ?? "IDLE",
      totalVectors: Number(d.total_vectors ?? 0),
      migrated: Number(d.migrated ?? 0),
      skipped: Number(d.skipped ?? 0),
      pctComplete: Number(d.pct_complete ?? 0),
      error: String(d.error ?? ""),
    };
  }

  async abortDrift(collection: string, opts?: RequestOptions): Promise<void> {
    await this.request("POST", this.col(collection, "/drift/abort"), {}, opts);
  }

  // ---- admin ----------------------------------------------------------

  async triggerBackup(opts?: RequestOptions): Promise<string> {
    return (await this.data<{ snapshot: string }>("POST", "/v1/admin/backup", {}, opts)).snapshot;
  }

  async listBackups(opts?: RequestOptions): Promise<string[]> {
    return (await this.data<{ snapshots?: string[] }>("GET", "/v1/admin/backups", undefined, opts)).snapshots ?? [];
  }

  async listBackupManifests(opts?: RequestOptions): Promise<BackupManifest[]> {
    const d = await this.data<{ manifests?: BackupManifest[] }>("GET", "/v1/admin/backups/manifests", undefined, opts);
    return d.manifests ?? [];
  }

  /** Point-in-time restore; the server must be restarted afterwards. */
  restoreBackup(snapshot: string, opts?: RequestOptions): Promise<Record<string, unknown>> {
    return this.data("POST", "/v1/admin/restore", { snapshot, confirm: true }, opts);
  }

  // ---- handles --------------------------------------------------------

  /** A handle bound to one collection. */
  collection(name: string): Collection {
    return new Collection(this, name);
  }
}

/** Collection-scoped view of a {@link TideVec} client. */
export class Collection {
  constructor(
    readonly client: TideVec,
    readonly name: string,
  ) {}

  info(opts?: RequestOptions) {
    return this.client.getCollection(this.name, opts);
  }
  exists(opts?: RequestOptions) {
    return this.client.hasCollection(this.name, opts);
  }
  drop(opts?: RequestOptions) {
    return this.client.dropCollection(this.name, opts);
  }
  count(opts?: CountOptions) {
    return this.client.count(this.name, opts);
  }
  upsert(vectors: VectorInput[], opts?: RequestOptions & { batchSize?: number }) {
    return this.client.upsert(this.name, vectors, opts);
  }
  delete(what: DeleteOptions | string[], opts?: RequestOptions) {
    return this.client.delete(this.name, what, opts);
  }
  search(vector: ArrayLike<number> | null, options?: SearchOptions) {
    return this.client.search(this.name, vector, options);
  }
  get(id: string, o?: GetOptions) {
    return this.client.get(this.name, id, o);
  }
  getMany(ids: string[], o?: GetOptions) {
    return this.client.getMany(this.name, ids, o);
  }
  scroll(o?: ScrollOptions) {
    return this.client.scroll(this.name, o);
  }
  ids(o?: Omit<ScrollOptions, "withVector" | "withPayload">) {
    return this.client.ids(this.name, o);
  }
  addEdges(edges: Edge[], opts?: RequestOptions) {
    return this.client.addEdges(this.name, edges, opts);
  }
  setTemporal(temporal: TemporalOptions, opts?: RequestOptions) {
    return this.client.setTemporal(this.name, temporal, opts);
  }
}
