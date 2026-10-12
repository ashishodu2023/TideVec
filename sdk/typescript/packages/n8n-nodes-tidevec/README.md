# n8n-nodes-tidevec

[n8n](https://n8n.io) community node for [TideVec](https://github.com/ashishodu2023/TideVec), the vector
database that knows when a fact was replaced. Upsert and search vectors from workflows; with supersession
on, a new version of a document hides the old one from search results.

Status: skeleton (0.3.0). Regular node with `usableAsTool`, so it can also be attached to an AI Agent node.
It is not a LangChain "Vector Store" sub-node yet.

## Install

In n8n: **Settings → Community Nodes → Install** and enter `n8n-nodes-tidevec`. For self-hosted
development:

```bash
cd sdk/typescript && npm install && npm run build --workspace tidevec --workspace n8n-nodes-tidevec
mkdir -p ~/.n8n/custom && cd ~/.n8n/custom && npm install /path/to/TideVec/sdk/typescript/packages/n8n-nodes-tidevec
```

## Credentials

**TideVec API**: server URL (default `http://localhost:6399`) and API key (sent as `X-Api-Key`; leave
empty for `--no-require-auth`). The credential test calls `GET /v1/collections`.

## Operations

| Resource | Operation | Parameters |
|---|---|---|
| Collection | Create | name, dimension, metric, half-life (days), supersession on/off, match field, threshold. Does nothing if the collection exists. |
| Collection | Get / List / Delete | name |
| Vector | Upsert | id, embedding (defaults to `{{ $json.embedding }}`), payload JSON, created at (ISO or epoch ms) |
| Vector | Search | embedding, top K, exclude superseded, filter `key=value`, temporal blend, superseded penalty |
| Vector | Delete | comma-separated ids and/or filter |

Embeddings come from an earlier node (an HTTP Request to an embeddings API, or an AI embeddings node).
Search emits one item per hit with `rank`, `id`, `score`, `vectorScore`, `temporalScore`, `createdAt`,
`payload` and `supersededBy`.

## Limitations

- One `key=value` filter per query (the 0.3 server's filter syntax).
- Depends on the `tidevec` npm package; n8n's verified-node programme requires zero runtime dependencies,
  so verification would need the client inlined or calls moved to `this.helpers.httpRequest`.

## Development

```bash
npm test --workspace n8n-nodes-tidevec   # spawns build*/tidevec-server or $TIDEVEC_SERVER_BIN
```
