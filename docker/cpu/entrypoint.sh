#!/bin/sh
# ================================================================
# TideVec entrypoint — expands env vars before exec'ing the server.
#
# Docker's exec-form CMD (["--port", "${TIDEVEC_PORT}"]) does NOT
# perform shell variable substitution — only shell-form CMD does.
# This script runs as the container's actual entrypoint so env vars
# set via `docker run -e TIDEVEC_PORT=...` or docker-compose are
# correctly expanded before reaching the binary.
# ================================================================
set -e

exec /usr/local/bin/tidevec-server \
    --host       "${TIDEVEC_HOST:-0.0.0.0}" \
    --port       "${TIDEVEC_PORT:-6399}" \
    --data-dir   "${TIDEVEC_DATA_DIR:-/data}" \
    --threads    "${TIDEVEC_THREADS:-8}" \
    --device     "${TIDEVEC_DEVICE:-cpu}"
