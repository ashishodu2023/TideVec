#!/usr/bin/env bash
# End-to-end production wiring test for tidevec-server
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
SERVER="$BUILD_DIR/tidevec-server"
PORT="${E2E_PORT:-16499}"
API_KEY="${E2E_API_KEY:-e2e_test_key}"
QUOTA_KEY="${E2E_QUOTA_KEY:-e2e_quota_key}"
HOST="127.0.0.1"
BASE="http://${HOST}:${PORT}"

PASS=0
FAIL=0
ERRORS=()

log() { echo "[e2e] $*"; }

pass() { PASS=$((PASS + 1)); log "PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); ERRORS+=("$1"); log "FAIL: $1"; }

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

curl_json() {
  local method="$1" path="$2" body="${3:-}" key_mode="${4:-with}"
  local api_key_val="${5:-$API_KEY}"
  local args=(-sS -w "\n%{http_code}" -X "$method" "${BASE}${path}")
  args+=(-H "Content-Type: application/json")
  if [[ "$key_mode" == "with" ]]; then
    args+=(-H "X-Api-Key: $api_key_val")
  elif [[ "$key_mode" == "bad" ]]; then
    args+=(-H "X-Api-Key: bad_key")
  fi
  if [[ -n "$body" ]]; then args+=(-d "$body"); fi
  curl "${args[@]}"
}

http_code() { echo "$1" | tail -n1; }
http_body() { echo "$1" | sed '$d'; }

wait_for_health() {
  local i
  for i in $(seq 1 60); do
    if curl -sf "${BASE}/health" >/dev/null 2>&1; then return 0; fi
    sleep 0.25
  done
  return 1
}

# --- Build if needed ---
if [[ ! -x "$SERVER" ]]; then
  log "Building tidevec-server..."
  mkdir -p "$BUILD_DIR"
  cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$BUILD_DIR" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)" --target tidevec-server
fi

DATA_DIR=$(mktemp -d "${TMPDIR:-/tmp}/tidevec-e2e.XXXXXX")
BACKUP_DIR="$DATA_DIR/backups"
mkdir -p "$BACKUP_DIR"

# Quota tenant (non-default id so quotas apply)
cat > "$DATA_DIR/tenants.json" << TENANTS_EOF
[
  {
    "id": "quota_tenant",
    "api_key": "${QUOTA_KEY}",
    "name": "Quota Test Tenant",
    "max_collections": 1,
    "max_vectors": 1000000,
    "enabled": true,
    "collection_prefixes": []
  }
]
TENANTS_EOF

log "Starting server on port $PORT (data: $DATA_DIR)"
unset TIDEVEC_REQUIRE_AUTH TIDEVEC_API_KEY || true
export TIDEVEC_REQUIRE_AUTH=1
"$SERVER" \
  --host "$HOST" \
  --port "$PORT" \
  --data-dir "$DATA_DIR" \
  --backup-dir "$BACKUP_DIR" \
  --require-auth \
  --api-key "$API_KEY" \
  --segment-store \
  --device cpu \
  --backup \
  --multi-tenant \
  --quiet \
  >"$DATA_DIR/server.log" 2>&1 &
SERVER_PID=$!

if ! wait_for_health; then
  fail "server did not become healthy"
  cat "$DATA_DIR/server.log" >&2 || true
  echo "PASSED=$PASS FAILED=$FAIL"
  exit 1
fi

# --- Tests ---
resp=$(curl_json GET /health "" none none)
code=$(http_code "$resp")
body=$(http_body "$resp")
if [[ "$code" == "200" ]] && echo "$body" | grep -q '"status"'; then
  pass "health"
else
  fail "health (HTTP $code)"
fi

resp=$(curl_json GET /v1/collections "" none none)
code=$(http_code "$resp")
if [[ "$code" == "401" ]]; then
  pass "auth 401 without key"
else
  fail "auth 401 without key (HTTP $code)"
fi

resp=$(curl_json GET /v1/collections "" none bad)
code=$(http_code "$resp")
if [[ "$code" == "401" ]]; then
  pass "auth 401 bad key"
else
  fail "auth 401 bad key (HTTP $code)"
fi

# Tenant quota before other collections (quota uses global registry size)
qcol1='quota_col_a'
qcol2='quota_col_b'
qb='{"name":"'"$qcol1"'","dim":4,"index_type":"flat","metric":"cosine"}'
resp=$(curl_json POST /v1/collections "$qb" with "$QUOTA_KEY")
code=$(http_code "$resp")
if [[ "$code" == "201" ]]; then
  pass "tenant quota setup (first collection)"
else
  fail "tenant quota setup (HTTP $code: $(http_body "$resp"))"
fi

qb2='{"name":"'"$qcol2"'","dim":4,"index_type":"flat","metric":"cosine"}'
resp=$(curl_json POST /v1/collections "$qb2" with "$QUOTA_KEY")
code=$(http_code "$resp")
body=$(http_body "$resp")
if [[ "$code" == "403" ]] && echo "$body" | grep -qi quota; then
  pass "tenant collection quota enforced"
else
  fail "tenant collection quota (HTTP $code: $body)"
fi

COL="e2e_prod_col"
create_body='{"name":"'"$COL"'","dim":4,"index_type":"flat","metric":"cosine","n_shards":1,"n_replicas":1,"write_quorum":1}'
resp=$(curl_json POST /v1/collections "$create_body")
code=$(http_code "$resp")
body=$(http_body "$resp")
if [[ "$code" == "201" ]] && echo "$body" | grep -q "$COL"; then
  pass "create collection"
else
  fail "create collection (HTTP $code: $body)"
fi

upsert_body='{"id":"v1","embedding":[1,0,0,0]}'
resp=$(curl_json POST "/v1/collections/${COL}/upsert" "$upsert_body")
code=$(http_code "$resp")
if [[ "$code" == "201" ]]; then
  pass "upsert"
else
  fail "upsert (HTTP $code: $(http_body "$resp"))"
fi

search_body='{"vector":[1,0,0,0],"top_k":2,"temporal_blend":0,"include_trace":true}'
resp=$(curl_json POST "/v1/collections/${COL}/search" "$search_body")
code=$(http_code "$resp")
body=$(http_body "$resp")
if [[ "$code" == "200" ]] && echo "$body" | grep -q '"trace"'; then
  pass "search with trace"
else
  fail "search with trace (HTTP $code: $body)"
fi

drift_body='{"new_dim":4,"reembed_url":"http://127.0.0.1:9999/reembed"}'
resp=$(curl_json POST "/v1/collections/${COL}/drift/start" "$drift_body")
code=$(http_code "$resp")
body=$(http_body "$resp")
if [[ "$code" == "403" ]] && echo "$body" | grep -qi "not allowed"; then
  pass "SSRF block on drift reembed_url"
else
  fail "SSRF block on drift (HTTP $code: $body)"
fi

resp=$(curl_json POST /v1/admin/backup "")
code=$(http_code "$resp")
body=$(http_body "$resp")
if [[ "$code" == "200" ]] && echo "$body" | grep -q snapshot; then
  pass "manual backup"
else
  fail "manual backup (HTTP $code: $body)"
fi

resp=$(curl_json GET /v1/admin/backups "")
code=$(http_code "$resp")
body=$(http_body "$resp")
if [[ "$code" == "200" ]] && echo "$body" | grep -q snapshots; then
  pass "list backups"
else
  fail "list backups (HTTP $code: $body)"
fi

resp=$(curl_json GET /v1/admin/backups/manifests "")
code=$(http_code "$resp")
if [[ "$code" == "200" ]]; then
  pass "backup manifests"
else
  fail "backup manifests (HTTP $code: $(http_body "$resp"))"
fi

# Python SDK smoke (optional)
SDK_SMOKE=skip
if command -v python3 >/dev/null 2>&1; then
  if E2E_PORT="$PORT" E2E_API_KEY="$API_KEY" PYTHONPATH="$ROOT/sdk/python" python3 - << 'PY' 2>"$DATA_DIR/sdk.err"; then
import os
from tidevec import TideVec
port = os.environ["E2E_PORT"]
key = os.environ["E2E_API_KEY"]
db = TideVec(f"127.0.0.1:{port}", api_key=key)
info = db.info()
assert info.get("name") == "TideVec"
db.create_collection("sdk_smoke", dim=4, index_type="flat", metric="cosine", n_shards=1)
db.upsert("sdk_smoke", [{"id": "s1", "embedding": [1.0, 0, 0, 0]}])
hits = db.search("sdk_smoke", [1.0, 0, 0, 0], top_k=1)
assert len(hits) >= 1
print("ok")
PY
    pass "python SDK smoke"
    SDK_SMOKE=ok
  else
    fail "python SDK smoke ($(cat "$DATA_DIR/sdk.err" 2>/dev/null | head -3))"
    SDK_SMOKE=fail
  fi
else
  log "SKIP: python3 not found"
fi

echo ""
echo "========== E2E SUMMARY =========="
echo "PASSED: $PASS"
echo "FAILED: $FAIL"
echo "Python SDK: $SDK_SMOKE"
if ((${#ERRORS[@]} > 0)); then
  echo "Errors:"
  for e in "${ERRORS[@]}"; do echo "  - $e"; done
fi
echo "Server log: $DATA_DIR/server.log"

[[ "$FAIL" -eq 0 ]]
