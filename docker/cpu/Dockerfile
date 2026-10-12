# ================================================================
# TideVec — CPU Image (Docker Hub: averm004/tidevec:cpu)
#
# Multi-arch: linux/amd64 · linux/arm64
# amd64 binaries target x86-64-v3 (Haswell or newer: AVX2 + FMA).
#
# Pull:
#   docker pull averm004/tidevec:latest
#   docker pull averm004/tidevec:cpu
#
# Run:
#   docker run -d \
#     -p 6399:6399 \
#     -v $(pwd)/data:/data \
#     averm004/tidevec:latest
#
# With custom config:
#   docker run -d \
#     -p 6399:6399 \
#     -v $(pwd)/data:/data \
#     -e TIDEVEC_THREADS=16 \
#     -e TIDEVEC_API_KEY=mysecretkey \
#     averm004/tidevec:latest
#
# Native TLS (PEM files mounted read-only):
#   docker run -d \
#     -p 6399:6399 \
#     -v $(pwd)/data:/data -v $(pwd)/tls:/tls:ro \
#     -e TIDEVEC_TLS_CERT=/tls/tls.crt -e TIDEVEC_TLS_KEY=/tls/tls.key \
#     averm004/tidevec:latest
# ================================================================

# ── Stage 1: Dependency cache ────────────────────────────────────
# Separate layer so apt packages are cached between code changes
FROM ubuntu:24.04 AS deps

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential g++-12 cmake ninja-build \
    libstdc++-12-dev \
    curl git pkg-config \
    ca-certificates libssl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# ── Stage 2: Builder ─────────────────────────────────────────────
FROM deps AS builder

ARG TARGETPLATFORM
ARG BUILDPLATFORM
ARG VERSION=0.3.0

WORKDIR /build

# Copy CMakeLists first — cache layer invalidates only if build config changes
COPY CMakeLists.txt .

# Copy source
COPY include/ include/
COPY src/     src/
COPY proto/   proto/

# Build with maximum optimisation for the target arch
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CXX_COMPILER=g++-12 \
    -DTIDEVEC_BUILD_TESTS=OFF \
    -DTIDEVEC_MARCH=x86-64-v3 \
    -DTIDEVEC_TLS=ON \
    -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" \
    -GNinja \
    && cmake --build build --target tidevec-server -j$(nproc) \
    && strip build/tidevec-server

# ── Stage 3: Runtime (minimal) ───────────────────────────────────
FROM ubuntu:24.04 AS runtime

ARG VERSION=0.3.0
ARG BUILD_DATE
ARG GIT_SHA

# OCI standard labels for Docker Hub
LABEL org.opencontainers.image.title="TideVec"
LABEL org.opencontainers.image.description="Temporally-aware causal vector database — CPU variant"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${GIT_SHA}"
LABEL org.opencontainers.image.source="https://github.com/ashishodu2023/TideVec"
LABEL org.opencontainers.image.url="https://tidevec.com"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.vendor="TideVec"
LABEL maintainer="contact@tidevec.com"

# Security: non-root user
RUN groupadd -r tidevec && useradd -r -g tidevec -s /sbin/nologin tidevec

# Minimal runtime deps (libssl3t64: OpenSSL 3 for native TLS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 libgcc-s1 libssl3t64 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Binary
COPY --from=builder /build/build/tidevec-server /usr/local/bin/tidevec-server
COPY docker/cpu/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Data dir
RUN mkdir -p /data /backups && chown tidevec:tidevec /data /backups

# Port: 6399 = REST
EXPOSE 6399

# Health check (https when TIDEVEC_TLS_CERT is set; the certificate is for
# the public name, so it is not verified against "localhost")
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD scheme=http; [ -z "$TIDEVEC_TLS_CERT" ] || scheme=https; \
        curl -sfk "$scheme://localhost:${TIDEVEC_PORT:-6399}/health" || exit 1

USER tidevec

# Snapshots tar up /data, so they must live outside it.
VOLUME ["/data", "/backups"]

# Environment variable defaults
ENV TIDEVEC_HOST=0.0.0.0
ENV TIDEVEC_PORT=6399
ENV TIDEVEC_DATA_DIR=/data
ENV TIDEVEC_THREADS=8
ENV TIDEVEC_DEVICE=cpu
ENV TIDEVEC_LOG_LEVEL=info

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
