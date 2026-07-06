#!/bin/sh
# ================================================================
# TideVec entrypoint — expands env vars before exec'ing the server.
# ================================================================
set -e

# Generate API key if not provided (auth on by default)
if [ -z "${TIDEVEC_API_KEY:-}" ]; then
    if command -v openssl >/dev/null 2>&1; then
        export TIDEVEC_API_KEY="tv_$(openssl rand -hex 16)"
    else
        export TIDEVEC_API_KEY="tv_dev_$(date +%s)"
    fi
    echo "[entrypoint] Generated TIDEVEC_API_KEY (save this): ${TIDEVEC_API_KEY}"
fi

EXTRA_ARGS=""
if [ "${TIDEVEC_REQUIRE_AUTH:-1}" != "0" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --require-auth"
fi
if [ "${TIDEVEC_ULTRA_DURABLE:-1}" != "0" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --ultra-durable"
fi
if [ "${TIDEVEC_SEGMENT_STORE:-1}" != "0" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --segment-store"
fi
if [ "${TIDEVEC_BACKUP:-0}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --backup"
fi
if [ -n "${TIDEVEC_OTEL_ENDPOINT:-}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --otel --otel-endpoint ${TIDEVEC_OTEL_ENDPOINT}"
fi
if [ -n "${TIDEVEC_TLS_CERT:-}" ] && [ -n "${TIDEVEC_TLS_KEY:-}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --tls-cert ${TIDEVEC_TLS_CERT} --tls-key ${TIDEVEC_TLS_KEY}"
fi

exec /usr/local/bin/tidevec-server \
    --host       "${TIDEVEC_HOST:-0.0.0.0}" \
    --port       "${TIDEVEC_PORT:-6399}" \
    --data-dir   "${TIDEVEC_DATA_DIR:-/data}" \
    --threads    "${TIDEVEC_THREADS:-8}" \
    --device     "${TIDEVEC_DEVICE:-auto}" \
    --api-key    "${TIDEVEC_API_KEY}" \
    --backup-dir "${TIDEVEC_BACKUP_DIR:-/data/backups}" \
    ${EXTRA_ARGS}
