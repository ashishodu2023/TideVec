# Retrieval, filters, as-of and hybrid search

Everything on this page is in the REST API ([reference](../api/rest.md)), the Python client and
embedded mode (`TideVec` and `Local` have the same methods), and the Go, Java and TypeScript clients.
The C++ SDK does not expose them ([details](../sdks/cpp.md)).

## Filters

`filter` is accepted by search, delete, scroll, ids and count. The short form is one equality on a
payload field:

```python
db.search("kb", q, top_k=10, filter="lang=en")
```

The structured form is a JSON object (a `dict` in Python):

```json
{"lang": "en"}                                   // equality
{"lang": {"$in": ["en", "de"]}}                  // $in / $nin
{"lang": {"$ne": "en"}}                          // $eq / $ne
{"price": {"$gte": 10, "$lt": 20}}               // $gt / $gte / $lt / $lte
{"published": {"$gte": "2024-01-01"}}            // ISO-8601 dates compare as time
{"created_at": {"$lt": 1700000000000}}           // the vector's own timestamp
{"_id": {"$in": ["a", "b"]}}                     // vector ids
{"tag": {"$exists": true}}
{"$and": [...]}, {"$or": [...]}, {"$not": {...}}
```

Several keys in one object are ANDed. An unknown operator is rejected with `400` (embedded mode
raises `TideVecError`).

### Payload indexes and query planning

Every payload key is indexed automatically: an inverted index for keyword values (equality, `$in`)
and a sorted index for values that parse as numbers or ISO-8601 dates (ranges); `created_at` is always
indexed. For each filtered search the planner estimates how many vectors match and picks one of:

| Plan | When |
|---|---|
| `graph` | the filter is not selective: HNSW traversal, skipping non-matching nodes (they still route the search) |
| `exact_index` | few vectors match and the indexes can enumerate them: exact search over just those |
| `exact_scan` | few match but the indexes cannot enumerate them (for example `$not`): exact scan |

Small result sets therefore keep full recall instead of an empty or short top-k. Indexes cost memory;
create a write-heavy collection with `payload_index=False` to turn them off (filters still work, by
scanning). The nightly CI job runs the planner against a brute-force oracle on 1M vectors
(`TIDEVEC_FILTER_BENCH_N=1000000`, test `filtered_recall_at_scale` in `tidevec_scale_tests`).

## Delete by filter

```python
db.delete("kb", ids=["a", "b"])                    # by id
db.delete("kb", filter="source=crm")               # or by filter; returns the number deleted
db.delete("kb", filter={"published": {"$lt": "2023-01-01"}})
```

REST: `POST /v1/collections/{name}/delete` with `ids` and/or `filter`.

## Reading data back: get, scroll, ids, count

Stored vectors can be read without a query vector:

```python
rec = db.get("kb", "refunds-2026")                 # Record or None
recs = db.get_many("kb", ["a", "b"], with_vector=True)
page = db.scroll("kb", limit=100, filter={"lang": "en"})
while page.next_cursor:                            # id order, stable cursor
    page = db.scroll("kb", limit=100, cursor=page.next_cursor, filter={"lang": "en"})
for rec in db.iterate("kb", filter="lang=en"):     # all pages
    ...
ids = db.list_ids("kb", filter={"tag": {"$exists": True}})
n = db.count("kb", filter={"lang": "en"})
```

REST: `GET /v1/collections/{name}/vectors/{id}`, `POST .../get`, `POST .../scroll`, `POST .../ids`,
`POST .../count`. LangChain `get_by_ids`, LlamaIndex node fetch and `AgentMemory.history` use these.

| | Python | Go | Java | TypeScript |
|---|---|---|---|---|
| one / many | `get`, `get_many` | `Get`, `GetMany` | `get` | `get`, `getMany` |
| pages | `scroll`, `iterate` | `Scroll` | `scroll` | `scrollPage` |
| ids | `list_ids` | `ListIDs` | `listIds` | `listIds` |
| count | `count(filter=, as_of=)` | `CountWhere` | `count` | `count` |

## Point-in-time reads (`as_of`)

`as_of` (epoch milliseconds, an ISO-8601 string, or a `datetime` in Python) evaluates a query as of
that moment: only vectors created at or before it are visible, recency is computed as if the query ran
then, and a supersession link only counts if the newer version already existed. Search, scroll, ids
and count accept it.

```python
db.search("policies", q, top_k=5, as_of="2025-07-01T00:00:00Z", superseded_penalty=1.0)
```

This answers "what would the agent have retrieved last month?", for audits and for reproducing an
old answer.

## Hybrid search (BM25)

Create the collection with `text_field`, the payload field to index for keyword search:

```python
db.create_collection("kb", dim=384, text_field="text")
db.search("kb", text="TV-4711")                              # keyword only
db.search("kb", q, text="TV-4711 shutdown")                  # hybrid, reciprocal-rank fusion
db.search("kb", q, text="TV-4711", hybrid="weighted", alpha=0.3)
```

Fusion is reciprocal-rank (`"rrf"`, default, constant `rrf_k=60`) or a weighted sum of min-max
normalised scores (`"weighted"`, `alpha` = weight of the vector side). Fused hits keep the vector
hit's recency and supersession fields and carry the BM25 score as `text_score`. Filters and `as_of`
apply to both sides. Exact tokens such as product codes and error IDs are where dense-only retrieval
tends to miss.

## Collection aliases

An alias is a second name for a collection that every collection route accepts. Repointing it is
atomic, so readers can be switched to a rebuilt collection (for example re-embedded with a model of a
different dimension) without downtime:

```python
db.create_collection("kb_v2", dim=1024)
# ... fill kb_v2 ...
db.set_alias("kb", "kb_v2")        # returns the collection "kb" pointed at before
db.list_aliases()                  # {"kb": "kb_v2"}
db.delete_alias("kb")
```

REST: `GET /v1/aliases`, `PUT /v1/aliases/{alias}`, `DELETE /v1/aliases/{alias}`. Aliases are
replicated in cluster mode and scoped per tenant ([auth](../operations/auth.md)).

## Compaction

Deletes and re-embeds leave tombstones in the vector, payload and text indexes. Once tombstones reach
`compaction_ratio` × live vectors (default 1.0, set per collection) the indexes are rebuilt **in the
background**: reads and writes continue, and writes made during the rebuild are replayed onto the new
index before it is swapped in. `compact(name, wait=False)` starts one by hand and
`compaction_status(name)` reports progress (REST: `POST`/`GET /v1/collections/{name}/compact`).
