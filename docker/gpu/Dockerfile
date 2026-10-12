# ================================================================
# TideVec — GPU Image (averm004/tidevec:gpu)
#
# Built with CUDA toolkit — real GPU kernels linked into tidevec-server.
# Runs on any machine with nvidia-container-toolkit installed.
#
# Pull:
#   docker pull averm004/tidevec:gpu
#
# Run (configured with the same TIDEVEC_* variables as the CPU image):
#   docker run -d --gpus all \
#     -p 6399:6399 -v $(pwd)/data:/data \
#     averm004/tidevec:gpu
# ================================================================

FROM nvidia/cuda:12.6.0-devel-ubuntu22.04 AS builder

ARG VERSION=0.3.0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build curl pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY CMakeLists.txt .
COPY include/ include/
COPY src/     src/
COPY proto/   proto/

RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 \
    -DCMAKE_CUDA_STANDARD=17 \
    -DCMAKE_CUDA_STANDARD_REQUIRED=ON \
    -DTIDEVEC_BUILD_TESTS=OFF \
    -DTIDEVEC_MARCH=x86-64-v3 \
    -DTIDEVEC_CUDA=ON \
    -DTIDEVEC_TLS=ON \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" \
    -GNinja \
    && cmake --build build --target tidevec-server -j$(nproc) \
    && strip build/tidevec-server

# ── Runtime on CUDA base ──────────────────────────────────────────
FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS runtime

ARG VERSION=0.3.0
ARG BUILD_DATE
ARG GIT_SHA

LABEL org.opencontainers.image.title="TideVec GPU"
LABEL org.opencontainers.image.description="TideVec with CUDA CAGRA kernels — GPU-accelerated search"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.source="https://github.com/ashishodu2023/TideVec"
LABEL org.opencontainers.image.url="https://tidevec.com"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL maintainer="contact@tidevec.com"

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN groupadd -r tidevec && useradd -r -g tidevec -s /sbin/nologin tidevec

RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 libgcc-s1 libssl3 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/build/tidevec-server /usr/local/bin/tidevec-server
COPY docker/cpu/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

RUN mkdir -p /data /backups && chown tidevec:tidevec /data /backups

EXPOSE 6399

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD scheme=http; [ -z "$TIDEVEC_TLS_CERT" ] || scheme=https; \
        curl -sfk "$scheme://localhost:${TIDEVEC_PORT:-6399}/health" || exit 1

USER tidevec
VOLUME ["/data", "/backups"]

# Same environment variables as the CPU image (see docker/cpu/entrypoint.sh).
ENV TIDEVEC_HOST=0.0.0.0
ENV TIDEVEC_PORT=6399
ENV TIDEVEC_DATA_DIR=/data
ENV TIDEVEC_DEVICE=gpu
ENV TIDEVEC_THREADS=16
ENV TIDEVEC_LOG_LEVEL=info

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
