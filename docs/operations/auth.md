# Auth, tenants and limits

## API keys

Routes under `/v1/` require an `X-Api-Key` header unless auth is disabled. `/health`, `/v1/info` and
`/metrics` are always open. The key comes from `--api-key` or `TIDEVEC_API_KEY`; the Docker image
generates a random one and prints it if neither is set. `TIDEVEC_REQUIRE_AUTH=0` (or
`--no-require-auth`) turns auth off, even when a key is configured (the server prints a warning and
every caller gets admin access); this is only appropriate on a trusted local machine. It cannot be
combined with `--multi-tenant`.

Clients send the key automatically when `TIDEVEC_API_KEY` is set in their environment (Python, Go,
Java) or when it is passed explicitly.

The configured key is the only admin key: change `--api-key` / `TIDEVEC_API_KEY` and restart, and
the old key stops working. (Before 0.3.0 the first key was also stored in `tenants.json` as a
`default` tenant and kept working; that entry is removed at startup.)

## Multi-tenant mode

With `--multi-tenant` (`TIDEVEC_MULTI_TENANT=1` in Docker), the server also accepts the keys listed
in `<data-dir>/tenants.json`, read at startup:

```json
[
  {"id": "acme", "api_key": "tv_...", "name": "Acme",
   "collection_prefixes": ["acme_"], "max_collections": 20, "max_vectors": 5000000,
   "enabled": true}
]
```

`id` and `api_key` are required; `default` is reserved for the admin key. `collection_prefixes`
limits which collections and aliases a tenant can see and use; it defaults to `["<id>_"]`, and
`[""]` grants every collection. `max_collections` (default 100) and `max_vectors` (default
10,000,000) are quotas on the tenant's own collections and vectors; `enabled: false` disables a key.
Entries that cannot be parsed are skipped with a message on stderr, so check the log after editing.

A tenant key:

- only sees its own collections in `GET /v1/collections` and `GET /v1/aliases`, and gets `403` on
  any other collection or alias;
- cannot call the admin and cluster endpoints (`/v1/admin/*`, `/v1/cluster/*`), which need the main
  key.

Tenant keys are only accepted with `--multi-tenant`. `/health`, `/v1/info` and `/metrics` stay open
and report server-wide totals (collection count, request counters), not per-tenant data.

## Rate limiting

Search, upsert and read-back routes are rate-limited per API key (or per client IP when no valid key is
sent) with a token bucket: `--rate-limit N` requests per second (default 100), burst 200. Over the
limit the server answers `429`; the Python, Go and Java clients retry with backoff up to
`max_retries` times. Behind a reverse proxy, keyless clients share the proxy's IP address.

## TLS

A server built with `-DTIDEVEC_TLS=ON` (needs OpenSSL 3) serves HTTPS with
`--tls-cert cert.pem --tls-key key.pem`; a build without TLS exits with an error if those flags are
given. The Docker images are built with TLS: set `TIDEVEC_TLS_CERT` and `TIDEVEC_TLS_KEY` to mounted
PEM files; in Kubernetes, `tls.enabled` and `tls.secretName` in the Helm chart. Cluster mode talks
plain HTTP between nodes. Alternatively, terminate TLS in a reverse proxy or load balancer:

- Docker Compose: `docker compose -f docker/docker-compose.yml --profile tls up` runs nginx with
  `docker/nginx/tidevec.conf` on port 443.
- Kubernetes: enable the chart's Ingress with a TLS secret.

Clients connect with `TideVec(host, tls=True)` (Python), `ClientConfig{TLS: true}` (Go) or
`.tls(true)` (Java).

## Cluster RPC

In [cluster mode](cluster.md) the `/raft/*` endpoints are protected by a shared secret
(`--cluster-secret` or `TIDEVEC_CLUSTER_SECRET`), separate from API keys. Without it the server logs a
warning and the Raft endpoints accept any caller.

## DriftBridge URL allowlist

[DriftBridge](../api/rest.md#post-v1-collections-name-drift-start) calls a re-embedding URL that
the client supplies, so the server only calls hosts in `--reembed-allow`. Loopback and private
addresses also need `--reembed-allow-private`, and cloud metadata endpoints are always blocked.
Only `http(s)://host[:port]/...` URLs with a plain host name or dotted IPv4 address are accepted.
When a migration starts, the host is resolved once and every address it resolves to is checked:
loopback, private, link-local and metadata addresses are refused (`403`) unless the host is listed in
`--reembed-allow` and `--reembed-allow-private` is set. All re-embedding calls of that migration go
to the address checked then, so a DNS change during the migration has no effect.
