# ================================================================
# TideVec — TPU Image (Docker Hub: averm004/tidevec:tpu)
#
# For Google Cloud TPU v4/v5e nodes via GKE.
# This image runs TideVec in CPU mode but with JAX/XLA installed
# so it can be extended for TPU acceleration.
#
# Full XLA integration requires the XLA client library to be
# built from source alongside TensorFlow/JAX — planned for v0.2.
#
# Pull:
#   docker pull averm004/tidevec:tpu
#
# Run:
#   docker run -d -p 6399:6399 -v $(pwd)/data:/data \
#     averm004/tidevec:tpu
# ================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────
FROM ubuntu:22.04 AS builder

ARG VERSION=0.1.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build \
    python3 python3-pip \
    curl git pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY CMakeLists.txt .
COPY include/ include/
COPY src/     src/
COPY proto/   proto/

# Build CPU-only server (XLA integration in v0.2)
RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=20 \
    -DTIDEVEC_BUILD_TESTS=OFF \
    -DTIDEVEC_XLA=OFF \
    -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" \
    -GNinja \
    && cmake --build build --target tidevec-server -j$(nproc) \
    && strip build/tidevec-server

# ── Stage 2: Runtime with JAX pre-installed ───────────────────────
FROM python:3.11-slim AS runtime

ARG VERSION=0.1.0
ARG BUILD_DATE
ARG GIT_SHA

LABEL org.opencontainers.image.title="TideVec TPU"
LABEL org.opencontainers.image.description="TideVec for Google Cloud TPU — XLA integration coming in v0.2"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${GIT_SHA}"
LABEL org.opencontainers.image.source="https://github.com/ashishodu2023/TideVec"
LABEL org.opencontainers.image.url="https://tidevec.com"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL maintainer="contact@tidevec.com"

RUN groupadd -r tidevec && useradd -r -g tidevec -s /sbin/nologin tidevec

RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 libgcc-s1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Install JAX CPU (lightweight — no TPU libs needed at image build time)
# TPU libs are injected at runtime by the GKE TPU node driver
RUN pip install --no-cache-dir \
    "jax[cpu]>=0.4.25" \
    "jaxlib>=0.4.25"

COPY --from=builder /build/build/tidevec-server /usr/local/bin/tidevec-server

RUN mkdir -p /data && chown tidevec:tidevec /data

EXPOSE 6399 6400

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:6399/health || exit 1

USER tidevec
VOLUME ["/data"]

ENV TIDEVEC_HOST=0.0.0.0
ENV TIDEVEC_PORT=6399
ENV TIDEVEC_DATA_DIR=/data
ENV TIDEVEC_THREADS=8
ENV TIDEVEC_DEVICE=cpu
ENV TPU_NAME=""
ENV CLOUD_TPU_TASK_ID=0

ENTRYPOINT ["/usr/local/bin/tidevec-server"]
CMD ["--host", "0.0.0.0", "--port", "6399", \
     "--data-dir", "/data", "--threads", "8"]
