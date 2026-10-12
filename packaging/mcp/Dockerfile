# Container for hosted MCP directories (Smithery). Runs tidevec-mcp over stdio.
FROM python:3.12-slim

ARG TIDEVEC_MCP_VERSION=0.3.0
RUN pip install --no-cache-dir "tidevec-mcp==${TIDEVEC_MCP_VERSION}" \
    && useradd --create-home --uid 1000 mcp

USER mcp
ENV TIDEVEC_MEMORY_PATH=/home/mcp/.tidevec/memory
ENTRYPOINT ["tidevec-mcp"]
