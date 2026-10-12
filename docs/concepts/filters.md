# Filters, as-of and hybrid search

## Shipped: `key=value` filters

Search and delete take a single equality filter on a payload field:

```python
db.search("kb", q, top_k=10, filter="lang=en")
db.delete("kb", filter="source=crm")          # returns the number deleted
```

Payload values are strings. The filter is applied **during graph traversal**: nodes that fail it are
not returned but still route the search, so selective filters keep their recall instead of emptying
the result list. All SDKs and both modes (server and embedded) support this form.

## In progress: richer filters, as-of queries and hybrid search

!!! warning "Unreleased: REST API only, subject to change"
    The features below are being added on `main` for the next release. They exist in the server's
    REST API (see the [reference](../api/rest.md)), but the Python, Go and Java SDKs and embedded mode
    do not expose them yet, and they are not covered by this release's changelog.
    <!-- TODO(docs): rewrite this section when the Phase 1 work lands in the SDKs. -->

**JSON filters.** `filter` may be a JSON object instead of a string:

```json
{"lang": "en"}                                   // equality
{"lang": {"$in": ["en", "de"]}}                  // $in / $nin
{"price": {"$gte": 10, "$lt": 20}}               // $gt / $gte / $lt / $lte
{"published": {"$gte": "2024-01-01"}}            // ISO-8601 dates compare as time
{"created_at": {"$lt": 1700000000000}}           // the vector's own timestamp
{"_id": {"$in": ["a", "b"]}}                     // vector ids
{"tag": {"$exists": true}}
{"$and": [...]}, {"$or": [...]}, {"$not": {...}}
```

**Point-in-time (`as_of`).** `as_of` (epoch milliseconds or an ISO-8601 string) restricts a query to
vectors created at or before that time, computes recency as if the query ran then, and only counts
supersession links whose newer version already existed. It is accepted by search, scroll, ids and
count.

**Hybrid search.** Collections created with a `text_field` keep a BM25 index over that payload field.
A search can send `text` alone (keyword only) or with `vector` (hybrid), fused either with
reciprocal-rank fusion (`"fusion": "rrf"`, the default) or a weighted sum of min-max-normalised
scores (`"fusion": "weighted"`), controlled by `alpha`. Fused hits keep the vector hit's recency and
supersession fields and carry the BM25 score separately.

**Reading data back.** `GET /v1/collections/{name}/vectors/{id}`, `POST .../get`, `POST .../scroll`,
`.../ids` and `POST .../count` fetch stored vectors without a query vector.

**Aliases.** `/v1/aliases/{alias}` gives a collection a second name that every collection route
accepts; repointing an alias is atomic, for cutting over to a rebuilt collection.
