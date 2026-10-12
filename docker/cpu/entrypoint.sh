#!/bin/sh
# ================================================================
# TideVec entrypoint — expands env vars before exec'ing the server.
# Arguments given to the container (docker run IMAGE --threads 32) are
# passed on after the ones derived from the environment, so they win.
# Shared by the cpu, gpu and tpu images.
# ================================================================
set -e

fail() {
    echo "[entrypoint] $*" >&2
    exit 1
}

# Generate an API key if auth is on and none was provided. The server reads
# TIDEVEC_API_KEY from its environment; it is not put on the command line,
# where any local user could read it from ps or /proc/<pid>/cmdline.
if [ "${TIDEVEC_REQUIRE_AUTH:-1}" != "0" ]; then
    if [ -z "${TIDEVEC_API_KEY:-}" ]; then
        TIDEVEC_API_KEY="tv_$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')"
        echo "[entrypoint] Generated TIDEVEC_API_KEY (save this): ${TIDEVEC_API_KEY}"
    fi
    export TIDEVEC_API_KEY
fi

# Arguments are collected as positional parameters so a value containing
# spaces stays one argument instead of being split into extra flags. The
# container's own arguments are kept in front for now and rotated to the
# end below.
n_user_args=$#
set -- "$@" \
    --host       "${TIDEVEC_HOST:-0.0.0.0}" \
    --port       "${TIDEVEC_PORT:-6399}" \
    --data-dir   "${TIDEVEC_DATA_DIR:-/data}" \
    --threads    "${TIDEVEC_THREADS:-8}" \
    --device     "${TIDEVEC_DEVICE:-auto}" \
    --backup-dir "${TIDEVEC_BACKUP_DIR:-/backups}"

if [ "${TIDEVEC_REQUIRE_AUTH:-1}" != "0" ]; then
    set -- "$@" --require-auth
else
    set -- "$@" --no-require-auth
fi
if [ "${TIDEVEC_ULTRA_DURABLE:-1}" != "0" ]; then
    set -- "$@" --ultra-durable
else
    set -- "$@" --no-ultra-durable
fi
if [ "${TIDEVEC_SEGMENT_STORE:-1}" != "0" ]; then
    set -- "$@" --segment-store
else
    set -- "$@" --no-segment-store
fi
if [ "${TIDEVEC_BACKUP:-0}" = "1" ]; then
    set -- "$@" --backup
fi
if [ -n "${TIDEVEC_BACKUP_INTERVAL_HOURS:-}" ]; then
    case "${TIDEVEC_BACKUP_INTERVAL_HOURS}" in
        ''|*[!0-9]*|0) fail "TIDEVEC_BACKUP_INTERVAL_HOURS must be a whole number of hours >= 1" ;;
    esac
    set -- "$@" --backup-interval-hours "${TIDEVEC_BACKUP_INTERVAL_HOURS}"
fi
# Uploads run the aws / gsutil CLIs, which these images do not ship.
if [ -n "${TIDEVEC_BACKUP_S3:-}" ]; then
    set -- "$@" --backup-s3 "${TIDEVEC_BACKUP_S3}"
fi
if [ -n "${TIDEVEC_BACKUP_GCS:-}" ]; then
    set -- "$@" --backup-gcs "${TIDEVEC_BACKUP_GCS}"
fi
if [ "${TIDEVEC_MULTI_TENANT:-0}" = "1" ]; then
    set -- "$@" --multi-tenant
fi
if [ -n "${TIDEVEC_OTEL_ENDPOINT:-}" ]; then
    set -- "$@" --otel --otel-endpoint "${TIDEVEC_OTEL_ENDPOINT}"
fi

# The server logs one line per request at "debug"/"info"; "warn"/"error"
# keep only startup output, warnings and errors. TIDEVEC_QUIET=1 is the
# same as "warn".
quiet=0
case "$(printf '%s' "${TIDEVEC_LOG_LEVEL:-info}" | tr '[:upper:]' '[:lower:]')" in
    debug|info) ;;
    warn|warning|error) quiet=1 ;;
    *) fail "TIDEVEC_LOG_LEVEL must be debug, info, warn or error (got '${TIDEVEC_LOG_LEVEL}')" ;;
esac
if [ "${TIDEVEC_QUIET:-0}" = "1" ] || [ "$quiet" = "1" ]; then
    set -- "$@" --quiet
fi

# Native TLS: PEM certificate (chain) and private key, both required.
if [ -n "${TIDEVEC_TLS_CERT:-}" ] || [ -n "${TIDEVEC_TLS_KEY:-}" ]; then
    [ -n "${TIDEVEC_TLS_CERT:-}" ] && [ -n "${TIDEVEC_TLS_KEY:-}" ] ||
        fail "TIDEVEC_TLS_CERT and TIDEVEC_TLS_KEY must be set together"
    [ -r "${TIDEVEC_TLS_CERT}" ] || fail "TIDEVEC_TLS_CERT is not readable: ${TIDEVEC_TLS_CERT}"
    [ -r "${TIDEVEC_TLS_KEY}" ] || fail "TIDEVEC_TLS_KEY is not readable: ${TIDEVEC_TLS_KEY}"
    set -- "$@" --tls-cert "${TIDEVEC_TLS_CERT}" --tls-key "${TIDEVEC_TLS_KEY}"
fi

if [ -n "${TIDEVEC_RATE_LIMIT:-}" ]; then
    set -- "$@" --rate-limit "${TIDEVEC_RATE_LIMIT}"
fi
if [ -n "${TIDEVEC_REEMBED_ALLOW:-}" ]; then
    set -- "$@" --reembed-allow "${TIDEVEC_REEMBED_ALLOW}"
fi
if [ "${TIDEVEC_REEMBED_ALLOW_PRIVATE:-0}" = "1" ]; then
    set -- "$@" --reembed-allow-private
fi
# Experimental Raft cluster mode; the secret is read from
# TIDEVEC_CLUSTER_SECRET by the server itself.
if [ -n "${TIDEVEC_NODE_ID:-}" ]; then
    set -- "$@" --node-id "${TIDEVEC_NODE_ID}" \
        --raft-peers "${TIDEVEC_RAFT_PEERS:?TIDEVEC_RAFT_PEERS is required with TIDEVEC_NODE_ID}"
fi

# Move the container's own arguments behind the generated ones.
i=0
while [ "$i" -lt "$n_user_args" ]; do
    set -- "$@" "$1"
    shift
    i=$((i + 1))
done

exec /usr/local/bin/tidevec-server "$@"
