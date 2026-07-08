# ================================================================
# TideVec — TPU Image (Docker Hub: averm004/tidevec:tpu)
#
# For Google Cloud TPU v4/v5e nodes via GKE.
# Attempts native XLA build when libxla_client is present in the
# TensorFlow devel base; otherwise builds CPU server. Runtime ships
# JAX for Cloud TPU orchestration.
#
# Pull:
#   docker pull averm004/tidevec:tpu
#
# Run on a Cloud TPU VM / GKE TPU node:
#   docker run -d -p 6399:6399 -v $(pwd)/data:/data \
#     -e TIDEVEC_DEVICE=tpu \
#     averm004/tidevec:tpu
# ================================================================

FROM tensorflow/tensorflow:2.15.0-devel AS builder

ARG VERSION=0.2.0

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*

COPY CMakeLists.txt .
COPY include/ include/
COPY src/     src/
COPY proto/   proto/

RUN XLA_INC=$(dirname "$(dirname "$(dirname "$(find /usr/local -name xla_builder.h 2>/dev/null | head -1)")")") && \
    XLA_LIB=$(dirname "$(find /usr/local -name 'libxla_client.so' 2>/dev/null | head -1)") && \
    if [ -n "$XLA_INC" ] && [ -f "$XLA_INC/xla/client/xla_builder.h" ] && \
       [ -n "$XLA_LIB" ] && [ -f "$XLA_LIB/libxla_client.so" ]; then \
      echo "Building with native XLA ($XLA_INC)"; \
      cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=20 \
        -DTIDEVEC_BUILD_TESTS=OFF \
        -DTIDEVEC_XLA=ON \
        -DXLA_INCLUDE_DIR="$XLA_INC" \
        -DXLA_LIB_DIR="$XLA_LIB" \
        -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" \
        -GNinja; \
    else \
      echo "XLA not found — building CPU server (JAX runtime still available)"; \
      cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=20 \
        -DTIDEVEC_BUILD_TESTS=OFF \
        -DTIDEVEC_XLA=OFF \
        -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" \
        -GNinja; \
    fi && \
    cmake --build build --target tidevec-server -j$(nproc) && \
    strip build/tidevec-server

# ── Runtime with JAX ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ARG VERSION=0.2.0
ARG BUILD_DATE
ARG GIT_SHA

LABEL org.opencontainers.image.title="TideVec TPU"
LABEL org.opencontainers.image.description="TideVec for Google Cloud TPU — XLA matmul + JAX runtime"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${GIT_SHA}"
LABEL org.opencontainers.image.source="https://github.com/ashishodu2023/TideVec"
LABEL org.opencontainers.image.url="https://gettidevec.com"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL maintainer="contact@tidevec.com"

RUN groupadd -r tidevec && useradd -r -g tidevec -s /sbin/nologin tidevec

RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 libgcc-s1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

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
ENV TIDEVEC_DEVICE=tpu
ENV TPU_NAME=""
ENV CLOUD_TPU_TASK_ID=0

ENTRYPOINT ["/usr/local/bin/tidevec-server"]
CMD ["--host", "0.0.0.0", "--port", "6399", \
     "--data-dir", "/data", "--threads", "8", "--device", "tpu"]
