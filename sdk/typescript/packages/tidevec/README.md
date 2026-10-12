# tidevec (TypeScript)

TypeScript/JavaScript client for [TideVec](https://github.com/ashishodu2023/TideVec), the vector
database that knows when a fact was replaced. Fetch-based, zero dependencies, typed, ESM and CommonJS;
runs on Node 18+, Deno, Bun, edge runtimes and browsers.

```bash
npm install tidevec
docker run -p 6399:6399 averm004/tidevec:0.3.0   # prints an API key on first start
```

## Quick start

```ts
import { TideVec, HalfLife } from "tidevec";

const db = new TideVec({ url: "localhost:6399", apiKey: process.env.TIDEVEC_API_KEY });

await db.createCollection("kb", {
  dim: 384,
  temporal: { halfLifeMs: HalfLife.ONE_MONTH, temporalBlend: 0.3 },
  // a new version of the same doc_id supersedes the old one at write time
  supersession: { matchField: "doc_id", similarityThreshold: 0.9 },
});

await db.upsert("kb", [
  { id: "refunds-v1", embedding: e1, payload: { doc_id: "refunds", text: "Refunds take 14 days" },
    createdAt: new Date("2025-01-10") },
  { id: "refunds-v2", embedding: e2, payload: { doc_id: "refunds", text: "Refunds take 3 days" } },
]);

const { hits } = await db.search("kb", queryVector, {
  topK: 5,
  temporalBlend: 0.3,       // recency weight for this query (0 = pure similarity)
  excludeSuperseded: true,  // or supersededPenalty: 1 to keep them at the bottom
  filter: { doc_id: "refunds" },
});
for (const h of hits) console.log(h.id, h.score, h.payload.text, h.supersededBy);
```

`apiKey` falls back to `$TIDEVEC_API_KEY` on Node. Pass `url` as `"host"`, `"host:port"` or a full
`http(s)://` URL.

## API

| Method | REST route |
|---|---|
| `health()`, `ping()`, `info()`, `metrics()`, `clusterStatus()` | `GET /health`, `/v1/info`, `/metrics`, `/v1/cluster/status` |
| `createCollection(name, opts)`, `ensureCollection`, `listCollections()`, `getCollection(name)`, `hasCollection`, `dropCollection`, `count` | `/v1/collections[/{name}]` |
| `setTemporal(name, {halfLifeMs, temporalBlend, stalenessThreshold})` | `PUT /v1/collections/{name}/temporal` |
| `upsert(name, vectors, {batchSize})` | `POST .../upsert` (batched, 500 per request by default) |
| `delete(name, ids)` / `delete(name, {ids, filter})` | `POST .../delete` |
| `search(name, vector, opts)` | `POST .../search` (dense, keyword with `vector = null`, or hybrid) |
| `get(name, id)`, `getMany(name, ids)` | `POST .../get` |
| `scroll(name, {filter, asOf, limit})` (async iterator), `scrollPage` | `POST .../scroll` |
| `ids(name, opts)` (async iterator), `listIds` | `POST .../ids` |
| `count(name, {filter, asOf})` | `POST .../count` (stats route when no filter) |
| `listAliases()`, `setAlias(alias, collection)`, `deleteAlias(alias)` | `/v1/aliases` |
| `addEdges(name, [{src, tgt, type, weight}])` | `POST .../edges` |
| `startDrift`, `driftStatus`, `abortDrift` | `.../drift/*` (DriftBridge re-embedding) |
| `triggerBackup`, `listBackups`, `listBackupManifests`, `restoreBackup` | `/v1/admin/*` |
| `request(method, path, body)` | any route, returns the raw JSON envelope |

`db.collection("kb")` returns a handle with the same methods bound to one collection.

### Search options

| Option | Meaning |
|---|---|
| `topK` | number of hits (default 10) |
| `temporalBlend` | weight of recency; omit for the collection default |
| `supersededPenalty` | multiply scores of superseded hits by `1 - penalty` |
| `excludeSuperseded` | drop superseded hits client-side (over-fetches to fill `topK`) |
| `mode` | `"vector_only"`, `"causal_expand"`, `"contradiction_check"`, `"entity_resolve"` |
| `causalHops` | hops for `causal_expand` |
| `filter` | `"key=value"` or a structured filter object (see below) |
| `asOf` | epoch ms, `Date` or ISO string: what was current at that time |
| `text`, `hybrid` | keyword query (needs `textField` on the collection) and fusion `{fusion: "rrf" \| "weighted", alpha, rrfK}` |
| `efSearch`, `metric`, `includeTrace`, `includeStalenessWarnings` | as in the REST API |
| `extra` | snake_case fields merged into the request body, for server features newer than the client |
| `signal` | `AbortSignal` |

Each hit has `id`, `score`, `vectorScore`, `temporalScore`, `createdAt`, `payload`, `supersededBy`,
`isSuperseded`, `contradictedBy`, `causalNeighbors`, `stalenessWarning`.

### Payloads and filters

Payloads are stored as string maps. Non-string values are JSON-encoded (`3` becomes `"3"`).

Filters are Mongo-style objects:

```ts
await db.search("kb", q, {
  filter: {
    $and: [{ lang: "en" }, { year: { $gte: 2020 } }, { tag: ["policy", "faq"] }],  // array = $in
  },
});
await db.count("kb", { filter: { created_at: { $gte: "2026-01-01T00:00:00Z" } } });
```

Operators: `$eq $ne $in $nin $gt $gte $lt $lte $exists`, combined with `$and $or $not`; range bounds are
numbers or ISO dates. A single-key equality such as `{ doc_id: "refunds" }` is sent as the legacy
`doc_id=refunds` string, so it also works against 0.3.0 servers; everything else needs a server with
structured filters, as do `get`, `scroll`, `ids`, filtered `count`, `asOf`, `text`/`hybrid` and aliases.

## Hybrid search and time travel

```ts
await db.createCollection("support", { dim: 384, textField: "text" });
await db.search("support", null, { text: "E1234" });                        // keyword only (BM25)
await db.search("support", q, { text: "E1234", hybrid: { fusion: "rrf" } }); // dense + keyword
await db.search("support", q, { asOf: "2026-03-01T00:00:00Z" });              // what an agent saw then
```

### Errors

All errors extend `TideVecError`: `TideVecConnectionError`, and `APIError` (`.status`, `.body`) with
subclasses `NotFoundError` (404), `UnauthorizedError` (401), `ForbiddenError` (403),
`RateLimitError` (429, retried `maxRetries` times first) and `NotLeaderError` (421, cluster mode; the
client follows the leader hint for up to `leaderWaitMs`).

## Agent memory

`AgentMemory` gives an agent versioned long-term memory. It uses the same storage layout as the Python
`AgentMemory` and the `tidevec-mcp` server, so memories written from Python, MCP clients and
TypeScript are interchangeable (when the same embedding model is used).

```ts
import { AgentMemory, TideVec } from "tidevec";

const mem = new AgentMemory(new TideVec(), { embed: (text) => myModel.embed(text) });
await mem.remember("User's favourite editor is Vim", { key: "user.editor" });
await mem.remember("User switched to Helix", { key: "user.editor" }); // supersedes Vim
await mem.recall("which editor does the user like?");                 // -> Helix only
await mem.history("user.editor");                                     // both, oldest first
await mem.forget({ key: "user.editor" });
```

`HashingEmbedder` is a dependency-free lexical embedder for tests and demos; use a real model in
production. For Vercel AI SDK tools built on this class see
[`@tidevec/ai-sdk`](../ai-sdk).

## Development

```bash
cd sdk/typescript && npm install
npm run build --workspace tidevec
npm test --workspace tidevec    # integration tests spawn build*/tidevec-server or $TIDEVEC_SERVER_BIN
```

Integration tests are skipped when no server binary is found; set `TIDEVEC_REQUIRE_SERVER=1` to make that
an error (CI).
