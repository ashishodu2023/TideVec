# Observability

## Health

`GET /health` (no auth) returns the version, number of collections and the server's mode flags
(`ultra_durable`, `segment_store`, `device`, `auth_required`, `multi_tenant`, `backup_enabled`). The
Docker image uses it as its `HEALTHCHECK`. `GET /v1/info` (no auth) returns the version and a feature
manifest; `GET /v1/cluster/status` reports Raft state in [cluster mode](cluster.md).

## Prometheus metrics

`GET /metrics` (no auth) serves Prometheus text format:

| Metric | Type |
|---|---|
| `tidevec_requests_total` | counter |
| `tidevec_search_requests_total` | counter |
| `tidevec_upsert_requests_total` | counter |
| `tidevec_errors_total` | counter |
| `tidevec_vectors_inserted_total` | counter |
| `tidevec_uptime_seconds` | gauge |

There are no latency histograms or per-collection metrics yet. Collection sizes and write/query counts
are available from `GET /v1/collections/{name}`. The compose `monitoring` profile includes a
Prometheus scrape config (`docker/prometheus.yml`).

## Logs

Each request is logged to stdout unless `--quiet` is set. With `--otel`, the server also writes
structured JSON lines for events (collection created, backup taken or restored, and one line per
search).

## Retrieval traces

`include_trace=true` on a search returns a trace with the strategy, latency, staleness warnings and
contradiction alerts in the response (`resp.trace` in Python).

### OTLP export

With `--otel-endpoint URL` (for example `http://otel-collector:4318/v1/traces`; a URL without a path
posts to `/v1/traces`), searches that request a trace are also sent to an OpenTelemetry collector as
OTLP/JSON spans named `tidevec.search`, with the collection, strategy, latency and recall estimate as
attributes. Spans are queued and posted by a background thread, so a slow or unreachable collector
does not delay searches; when the queue (1024 spans) is full, spans are dropped. The first failed
export is logged to stderr. `https://` endpoints need a `-DTIDEVEC_TLS=ON` build.
