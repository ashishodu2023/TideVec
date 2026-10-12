/** Stored payloads are flat string -> string maps. */
export type Payload = Record<string, string>;

/**
 * Payload accepted on write. Non-string values are JSON-encoded
 * (see {@link payloadStr}), so filters must use the same encoding.
 */
export type PayloadInput = Record<string, unknown>;

export type SearchMode =
  | "vector_only"
  | "causal_expand"
  | "contradiction_check"
  | "entity_resolve";

export type EdgeType =
  | "CAUSES"
  | "CONTRADICTS"
  | "UPDATES"
  | "RELATED_TO"
  | "ENTITY_OF"
  | "SUPPORTS"
  | "SUPERSEDED_BY";

export type Metric = "cosine" | "l2" | "dot" | (string & {});

/** Operators of a structured filter condition on one payload key (or `created_at`). */
export interface FilterCondition {
  $eq?: unknown;
  $ne?: unknown;
  $in?: unknown[];
  $nin?: unknown[];
  $gt?: number | string;
  $gte?: number | string;
  $lt?: number | string;
  $lte?: number | string;
  $exists?: boolean;
}

/**
 * Structured filter: `{ lang: "en", year: { $gte: 2020 }, tag: ["a", "b"] }`,
 * combined with `$and` / `$or` / `$not`. Range bounds may be numbers or
 * ISO-8601 dates; `created_at` filters on the vector timestamp.
 */
export interface FilterObject {
  $and?: FilterObject[];
  $or?: FilterObject[];
  $not?: FilterObject;
  [key: string]: unknown;
}

/**
 * Payload filter.
 *
 * - a string is sent verbatim (`"key=value"`, understood by every server)
 * - a single-key equality object such as `{ doc_id: "refunds" }` is sent
 *   as `"doc_id=refunds"`, so it also works on 0.3.0 servers
 * - any other object is sent as a structured filter (servers with JSON filters)
 */
export type Filter = string | FilterObject;

/** `as_of` time travel: epoch milliseconds, a Date or an ISO-8601 string. */
export type AsOf = number | Date | string;

export interface HybridOptions {
  /** `"rrf"` (reciprocal rank fusion, default) or `"weighted"`. */
  fusion?: "rrf" | "weighted";
  /** Weight of the dense score for weighted fusion. */
  alpha?: number;
  rrfK?: number;
  prefetch?: number;
}

export interface InlineEdge {
  targetId: string;
  type: EdgeType;
  weight?: number;
}

export interface VectorInput {
  id: string;
  embedding: ArrayLike<number>;
  payload?: PayloadInput;
  /** Milliseconds since the epoch; defaults to the server's clock. Accepts a Date. */
  createdAt?: number | Date;
  validFrom?: number | Date;
  validUntil?: number | Date;
  ttlSeconds?: number;
  edges?: InlineEdge[];
}

export interface Edge {
  src: string;
  tgt: string;
  type: EdgeType;
  weight?: number;
}

export interface SupersessionOptions {
  enabled?: boolean;
  /** vector_score ((1 + cos) / 2 for cosine) at which a new vector supersedes an old one. */
  similarityThreshold?: number;
  /** Only supersede when this payload field is equal (e.g. `"doc_id"`). */
  matchField?: string;
  candidates?: number;
}

export interface TemporalOptions {
  halfLifeMs?: number;
  temporalBlend?: number;
  stalenessThreshold?: number;
  temporalCutoff?: number;
}

export interface AgentOptions {
  enabled?: boolean;
  agentId?: string;
  memoryTtlSeconds?: number;
  dedupThreshold?: number;
  crossAgentReadable?: string[];
}

export interface CreateCollectionOptions {
  dim: number;
  indexType?: string;
  metric?: Metric;
  nShards?: number;
  nReplicas?: number;
  writeQuorum?: number;
  temporal?: TemporalOptions;
  /** `true` enables supersession with server defaults. */
  supersession?: boolean | SupersessionOptions;
  agent?: AgentOptions;
  /** Payload key indexed for keyword (BM25) and hybrid search. */
  textField?: string;
  /** Build payload indexes for filters (server default true). */
  payloadIndex?: boolean;
}

export interface SearchOptions {
  topK?: number;
  /** Weight of recency in the score; omit to use the collection setting, 0 = pure similarity. */
  temporalBlend?: number;
  /** Score multiplier (1 - penalty) for hits a newer version supersedes; 1 pushes them to the bottom. */
  supersededPenalty?: number;
  /** Drop superseded hits client-side (over-fetches to still return `topK`). */
  excludeSuperseded?: boolean;
  mode?: SearchMode;
  causalHops?: number;
  filter?: Filter;
  metric?: Metric;
  efSearch?: number;
  includeTrace?: boolean;
  includeStalenessWarnings?: boolean;
  /** Return what was current at this time (vectors created before it, supersession state then). */
  asOf?: AsOf;
  /**
   * Keyword query for BM25 search on the collection's `textField`. Combined
   * with a vector it runs hybrid search; pass `null` as the vector for keyword-only.
   */
  text?: string;
  hybrid?: HybridOptions;
  /**
   * Extra snake_case fields merged into the request body, for server
   * features newer than this client.
   */
  extra?: Record<string, unknown>;
  signal?: AbortSignal;
}

export interface StoredVector {
  id: string;
  createdAt: number;
  validFrom: number;
  validUntil?: number;
  payload: Payload;
  embedding?: number[];
  edges: { targetId: string; type: EdgeType; weight: number; createdAt?: number }[];
  /** Newer versions that replace this one. */
  supersededBy: string[];
  /** Older versions this one replaces. */
  supersedes: string[];
  isSuperseded: boolean;
}

export interface GetOptions {
  withVector?: boolean;
  withPayload?: boolean;
  signal?: AbortSignal;
}

export interface ScrollOptions extends GetOptions {
  filter?: Filter;
  asOf?: AsOf;
  /** Page size (server default 100 for vectors, 1000 for ids). */
  limit?: number;
  /** Resume from a `nextCursor`. */
  cursor?: string | null;
}

export interface ScrollPage<T> {
  items: T[];
  nextCursor: string | null;
}

export interface Alias {
  alias: string;
  collection: string;
}

export interface SearchHit {
  id: string;
  score: number;
  vectorScore: number;
  temporalScore: number;
  createdAt: number;
  payload: Payload;
  stalenessWarning: boolean;
  stalenessReason: string;
  causalNeighbors: string[];
  contradictedBy: string[];
  supersededBy: string[];
  /** True when a newer version supersedes this hit. */
  isSuperseded: boolean;
}

export interface RetrievalTrace {
  query_id?: string;
  strategy?: string;
  latency_ms?: number;
  [key: string]: unknown;
}

export interface SearchResponse {
  hits: SearchHit[];
  trace?: RetrievalTrace;
}

export interface CollectionInfo {
  name: string;
  nVectors: number;
  nShards: number;
  dim?: number;
  indexType?: string;
  metric?: string;
  backend?: string;
  totalWrites?: number;
  totalQueries?: number;
  supersession?: Record<string, unknown>;
  textField?: string;
  aliases?: string[];
}

export interface DriftStartOptions {
  reembedUrl: string;
  M?: number;
  efConstruction?: number;
  newDim?: number;
  temporal?: TemporalOptions;
}

export type DriftPhase = "IDLE" | "MIGRATING" | "SWAPPING" | "COMPLETE" | "FAILED" | "UNKNOWN";

export interface DriftStatus {
  collection: string;
  phase: DriftPhase;
  totalVectors: number;
  migrated: number;
  skipped: number;
  pctComplete: number;
  error: string;
}

export interface HealthResponse {
  status: string;
  version: string;
  collections: number;
  timestamp_ms: number;
  [key: string]: unknown;
}

export interface BackupManifest {
  snapshot: string;
  created_at: number;
  data_dir: string;
  s3_uri?: string;
  gcs_uri?: string;
}

/** Temporal decay half-life presets in milliseconds. */
export const HalfLife = {
  ONE_HOUR: 3_600_000,
  ONE_DAY: 86_400_000,
  ONE_WEEK: 604_800_000,
  ONE_MONTH: 2_592_000_000,
  ONE_YEAR: 31_536_000_000,
} as const;
