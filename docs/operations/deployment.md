# Deployment

## Docker

```bash
export TIDEVEC_API_KEY="$(openssl rand -hex 16)"
docker run -d --name tidevec -p 6399:6399 -e TIDEVEC_API_KEY \
  -v tidevec_data:/data -v tidevec_backups:/backups \
  averm004/tidevec:0.3.0
```

### Image tags

| Tag | Contents |
|---|---|
| `X.Y.Z`, `X.Y.Z-cpu` | a release, CPU image (linux/amd64 + linux/arm64) |
| `X.Y`, `X.Y-cpu` | latest patch of a minor release |
| `latest`, `cpu` | latest stable release (not pre-releases) |
| `X.Y.Z-gpu`, `X.Y-gpu`, `gpu` | CUDA image (amd64), same release |
| `X.Y.Z-tpu`, `X.Y-tpu`, `tpu` | TPU image (amd64), same release |
| `main`, `sha-<commit>` | builds of the `main` branch; not releases |

The amd64 CPU binary targets **x86-64-v3** (Haswell or newer: AVX2 and FMA). Older x86 CPUs need a
source build with `-DTIDEVEC_MARCH=x86-64` (or empty).

!!! warning "GPU and TPU images"
    The GPU path is an independent CUDA implementation that has not been validated on GPU hardware in
    CI, and the TPU image contains the XLA stub only. They use the same entrypoint and environment
    variables as the CPU image (`TIDEVEC_DEVICE` defaults to `gpu` / `tpu`).

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `TIDEVEC_API_KEY` | random, printed at startup | API key for `X-Api-Key` |
| `TIDEVEC_REQUIRE_AUTH` | `1` | `0` disables auth, even if `TIDEVEC_API_KEY` is set (local development only; no key is generated) |
| `TIDEVEC_ULTRA_DURABLE` | `1` | Reed-Solomon segments, group commit, health monitor ([durability](durability.md)) |
| `TIDEVEC_SEGMENT_STORE` | `1` | disk-backed vector segments |
| `TIDEVEC_BACKUP` | `0` | `1` takes a snapshot periodically ([backups](backups.md)) |
| `TIDEVEC_BACKUP_INTERVAL_HOURS` | `6` | hours between snapshots |
| `TIDEVEC_BACKUP_DIR` | `/backups` | where snapshots are written; keep it outside `/data` |
| `TIDEVEC_BACKUP_S3` / `TIDEVEC_BACKUP_GCS` | unset | also upload snapshots (needs `aws` / `gsutil`, not in the image) |
| `TIDEVEC_MULTI_TENANT` | `0` | `1` enables per-tenant keys from `/data/tenants.json` ([auth](auth.md)) |
| `TIDEVEC_RATE_LIMIT` | 100 | requests per second per API key (or client IP), burst 200 |
| `TIDEVEC_OTEL_ENDPOINT` | unset | enables trace export ([observability](observability.md)) |
| `TIDEVEC_REEMBED_ALLOW` | unset | comma-separated hosts DriftBridge may call |
| `TIDEVEC_REEMBED_ALLOW_PRIVATE` | `0` | `1` also allows loopback and private addresses |
| `TIDEVEC_NODE_ID`, `TIDEVEC_RAFT_PEERS`, `TIDEVEC_CLUSTER_SECRET` | unset | [cluster mode](cluster.md) (experimental) |
| `TIDEVEC_THREADS` | 8 | HTTP worker threads |
| `TIDEVEC_LOG_LEVEL` | `info` | `debug` / `info` log every request; `warn` / `error` only startup output, warnings and errors |
| `TIDEVEC_QUIET` | `0` | `1` is the same as `TIDEVEC_LOG_LEVEL=warn` |
| `TIDEVEC_TLS_CERT`, `TIDEVEC_TLS_KEY` | unset | PEM certificate (chain) and key: serve HTTPS ([TLS](auth.md#tls)); both or neither |
| `TIDEVEC_HOST`, `TIDEVEC_PORT`, `TIDEVEC_DATA_DIR` | `0.0.0.0`, `6399`, `/data` | |

Arguments after the image name are passed to the server after the ones built from these variables,
so they take precedence: `docker run averm004/tidevec --threads 32`.

```bash
docker run -d -p 6399:6399 -v "$PWD/data:/data" -v "$PWD/tls:/tls:ro" \
  -e TIDEVEC_TLS_CERT=/tls/tls.crt -e TIDEVEC_TLS_KEY=/tls/tls.key averm004/tidevec
```

### Compose

[`docker/docker-compose.yml`](https://github.com/ashishodu2023/TideVec/blob/main/docker/docker-compose.yml)
runs the CPU server with named volumes for data and backups. Optional profiles:

- `--profile tls`: nginx on port 443 in front of the server, as an alternative to native TLS (put `tidevec.crt` / `tidevec.key` in
  `docker/nginx/certs/`). Port 6399 stays published too; remove it from the `tidevec` service if only
  TLS traffic should reach the host.
- `--profile monitoring`: Prometheus (scraping `/metrics`), Grafana (no dashboards provisioned) and
  Jaeger.
- `--profile gpu`: the GPU image on port 6401.

## Kubernetes

A Helm chart lives in
[`deploy/helm/tidevec`](https://github.com/ashishodu2023/TideVec/tree/main/deploy/helm/tidevec)
(StatefulSet with a persistent volume, Secret for the API key, optional Ingress). It defaults to one
replica and writes backups to `/backups`, outside the data volume. `replicaCount > 1` is refused
unless `cluster.enabled=true` (experimental Raft, odd `replicaCount >= 3`); see
[Cluster mode](cluster.md) for what that mode does and does not guarantee.

`tls.enabled=true` with `tls.secretName` (a `kubernetes.io/tls` Secret, e.g. from cert-manager)
makes the server itself serve HTTPS; probes, the ServiceMonitor and `helm test` switch to HTTPS. It
cannot be combined with `cluster.enabled` (nodes talk plain HTTP to each other). Behind an Ingress,
tell the controller the backend speaks HTTPS (ingress-nginx:
`nginx.ingress.kubernetes.io/backend-protocol: HTTPS`). `config.logLevel` and
`backup.intervalHours` map to the variables above.

<!-- TODO(docs): the chart is still changing on main. Document its values here once it is final. -->

## Binary

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j"$(nproc)" --target tidevec-server
./build-release/tidevec-server --data-dir ./data --api-key "$TIDEVEC_API_KEY"
```

| Flag | Default | |
|---|---|---|
| `--host`, `--port` | `0.0.0.0`, `6399` | |
| `--data-dir` | `./tidevec_data` | |
| `--threads` | 8 | HTTP worker threads |
| `--api-key KEY` | `$TIDEVEC_API_KEY` | |
| `--require-auth` / `--no-require-auth` | required (`TIDEVEC_REQUIRE_AUTH=0` disables) | with auth required and no key, every `/v1` call is rejected; `--no-require-auth` turns auth off even if a key is set (warning at startup) and cannot be combined with `--multi-tenant` |
| `--ultra-durable` / `--no-ultra-durable` | on (forced off in cluster mode) | |
| `--segment-store` / `--no-segment-store` | on | |
| `--backup`, `--backup-dir`, `--backup-s3`, `--backup-gcs` | off, `./backups` | |
| `--backup-interval-hours N` | 6 | hours between snapshots |
| `--rate-limit N` | 100 | requests per second per client |
| `--multi-tenant` | off | |
| `--otel`, `--otel-endpoint URL` | off | |
| `--reembed-allow h1,h2`, `--reembed-allow-private` | none | DriftBridge URL allowlist |
| `--device auto\|cpu\|gpu\|tpu` | `auto` | |
| `--quiet` | | no per-request log lines |
| `--node-id`, `--raft-peers`, `--cluster-secret`, ... | | [cluster mode](cluster.md) |

`--tls-cert cert.pem --tls-key key.pem` serve HTTPS. They need a build with `-DTIDEVEC_TLS=ON`
(OpenSSL 3); a build without TLS exits with an error rather than serving plain HTTP.

## Data directory

The data directory holds `registry.json` (collections), a write-ahead log and segments per shard,
`tenants.json` (API keys; protect it like a secrets file) and, in cluster mode, `raft/`. Only one
server process may use a data directory at a time.
