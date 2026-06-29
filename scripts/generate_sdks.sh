#!/usr/bin/env bash
# ================================================================
# generate_sdks.sh — regenerate all language SDKs from proto
#
# Run after any change to proto/tidevec.proto
# Requires: python3, grpcio-tools, (optionally) protoc-gen-go, protoc-gen-java
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
PROTO="$ROOT/proto/tidevec.proto"

echo "=== TideVec SDK Generator ==="
echo "Proto: $PROTO"
echo ""

# ---- Python (grpcio-tools) ------------------------------------
echo "→ Python..."
pip install grpcio-tools --quiet
python3 -m grpc_tools.protoc \
    -I "$ROOT/proto" \
    --python_out="$ROOT/sdk/python/tidevec" \
    --grpc_python_out="$ROOT/sdk/python/tidevec" \
    "$PROTO"
echo "  ✅ sdk/python/tidevec/tidevec_pb2{,_grpc}.py"

# ---- Go (protoc-gen-go) ---------------------------------------
echo "→ Go..."
if command -v protoc &> /dev/null && command -v protoc-gen-go &> /dev/null; then
    mkdir -p "$ROOT/sdk/go/tidevec"
    protoc \
        -I "$ROOT/proto" \
        --go_out="$ROOT/sdk/go" \
        --go_opt=paths=source_relative \
        --go-grpc_out="$ROOT/sdk/go" \
        --go-grpc_opt=paths=source_relative \
        "$PROTO"
    echo "  ✅ sdk/go/tidevec/tidevec.pb.go"
else
    echo "  ⚠️  protoc / protoc-gen-go not found. Install with:"
    echo "     go install google.golang.org/protobuf/cmd/protoc-gen-go@latest"
    echo "     go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest"
    echo "  Skipping Go generation."
fi

# ---- Java (protoc + grpc-java plugin) -------------------------
echo "→ Java..."
if command -v mvn &> /dev/null; then
    cd "$ROOT/sdk/java"
    mvn generate-sources -q
    echo "  ✅ target/generated-sources/protobuf/"
    cd "$ROOT"
else
    echo "  ⚠️  Maven not found. Run 'mvn generate-sources' in sdk/java/ manually."
fi

# ---- C++ (protoc) ---------------------------------------------
echo "→ C++..."
if command -v protoc &> /dev/null; then
    mkdir -p "$ROOT/include/tidevec/api/proto"
    protoc \
        -I "$ROOT/proto" \
        --cpp_out="$ROOT/include/tidevec/api/proto" \
        --grpc_out="$ROOT/include/tidevec/api/proto" \
        --plugin=protoc-gen-grpc="$(which grpc_cpp_plugin 2>/dev/null || echo '')" \
        "$PROTO" 2>/dev/null || \
    echo "  ⚠️  grpc_cpp_plugin not found — C++ proto stubs not generated."
    echo "  ✅ include/tidevec/api/proto/tidevec.pb.{h,cc}"
else
    echo "  ⚠️  protoc not found. Install: apt install protobuf-compiler"
fi

echo ""
echo "=== SDK generation complete ==="
