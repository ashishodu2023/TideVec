"""
Multi-Raft sharding (EXPERIMENTAL): two independent 3-node Raft groups
behind tidevec-router. Ids hash (fmix64 of FNV-1a 64) to the group owning that
range; search and count scatter-gather; DDL is broadcast.

    TIDEVEC_SERVER_BIN=build-raft/tidevec-server pytest tests/cluster/test_multi_raft.py
"""
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chaos"))
from tvcluster import LocalCluster, call, free_ports, server_binary  # noqa: E402

BIN = server_binary()


def router_binary():
    if os.environ.get("TIDEVEC_ROUTER_BIN"):
        return os.environ["TIDEVEC_ROUTER_BIN"]
    if BIN:
        p = Path(BIN).with_name("tidevec-router")
        if p.exists():
            return str(p)
    return None


ROUTER = router_binary()
pytestmark = pytest.mark.skipif(BIN is None or ROUTER is None,
                                reason="tidevec-server / tidevec-router not built")
DIM = 8
U64 = 2**64 - 1


def shard_hash(s: str) -> int:
    h = 1469598103934665603
    for b in s.encode():
        h ^= b
        h = (h * 1099511628211) & U64
    h ^= h >> 33
    h = (h * 0xff51afd7ed558ccd) & U64
    h ^= h >> 33
    h = (h * 0xc4ceb9fe1a85ec53) & U64
    return h ^ (h >> 33)


def owner(id_: str, n_groups: int) -> int:
    step = U64 // n_groups
    h = shard_hash(id_)
    for g in range(n_groups):
        hi = U64 if g == n_groups - 1 else (g + 1) * step
        if h <= hi:
            return g
    raise AssertionError


@pytest.fixture
def sharded(tmp_path):
    groups = [LocalCluster(BIN, tmp_path / f"g{i}", n=3, snapshot_threshold=100) for i in range(2)]
    for g in groups:
        for i in g.initial_ids:
            g.start(i)
    for g in groups:
        g.wait_leader(timeout=30)
    port = free_ports(1)[0]
    spec = ";".join(",".join(n.url for n in g.nodes.values()) for g in groups)
    log = open(tmp_path / "router.log", "ab")
    proc = subprocess.Popen([ROUTER, "--port", str(port), "--groups", spec, "--timeout-ms", "10000"],
                            stdout=log, stderr=log)
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 10
    while call(url, "GET", "/health", timeout=1)[0] != 200:
        assert time.time() < deadline and proc.poll() is None, "router did not start"
        time.sleep(0.05)
    yield url, groups
    proc.terminate()
    proc.wait(timeout=10)
    for g in groups:
        g.stop_all()


def vec(rng):
    v = [rng.uniform(-1, 1) for _ in range(DIM)]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def test_router_shards_writes_and_merges_reads(sharded):
    url, groups = sharded
    rng = random.Random(7)
    st, j = call(url, "POST", "/v1/collections", {"name": "docs", "dim": DIM})
    assert st == 201, j
    for g in groups:   # broadcast DDL
        assert g.request(g.leader(), "GET", "/v1/collections/docs")[0] == 200
    st, _ = call(url, "POST", "/v1/collections", {"name": "docs", "dim": DIM})
    assert st == 409

    vecs = {f"doc-{i}": vec(rng) for i in range(200)}
    items = list(vecs.items())
    for b in range(0, len(items), 25):
        body = {"vectors": [{"id": k, "embedding": v, "payload": {"n": k}} for k, v in items[b:b + 25]]}
        st, j = call(url, "POST", "/v1/collections/docs/upsert", body, timeout=20)
        assert st == 201, j
        assert j["data"]["inserted"] == 25

    # Each id lives only on the group that owns its hash range.
    per_group = [0, 0]
    for k in vecs:
        per_group[owner(k, 2)] += 1
    for gi, g in enumerate(groups):
        st, j = g.leader_request("POST", "/v1/collections/docs/count", {})
        assert st == 200 and j["data"]["count"] == per_group[gi], (gi, j, per_group)
    assert min(per_group) > 50
    k = next(iter(vecs))
    other = groups[1 - owner(k, 2)]
    assert other.leader_request("GET", f"/v1/collections/docs/vectors/{k}")[0] == 404

    st, j = call(url, "POST", "/v1/collections/docs/count", {})
    assert st == 200 and j["data"]["count"] == 200
    st, j = call(url, "GET", "/v1/collections/docs/vectors/doc-42?consistency=linearizable")
    assert st == 200 and j["data"]["payload"]["n"] == "doc-42"
    st, j = call(url, "GET", "/v1/collections/docs/vectors/missing")
    assert st == 404

    # Scatter-gather search: the exact vector from either group ranks first.
    for k in ("doc-3", "doc-150", "doc-77"):
        st, j = call(url, "POST", "/v1/collections/docs/search", {"vector": vecs[k], "top_k": 5})
        assert st == 200, j
        res = j["data"]["results"]
        assert res[0]["id"] == k and len(res) == 5
        assert [r["score"] for r in res] == sorted((r["score"] for r in res), reverse=True)

    st, j = call(url, "POST", "/v1/collections/docs/delete", {"ids": ["doc-1", "doc-2", "doc-3"]})
    assert st == 200 and j["data"]["deleted"] == 3
    assert call(url, "POST", "/v1/collections/docs/count", {})[1]["data"]["count"] == 197

    st, j = call(url, "GET", "/v1/cluster/placement")
    assert st == 200 and len(j["data"]["groups"]) == 2

    assert call(url, "DELETE", "/v1/collections/docs")[0] == 200
    for g in groups:
        assert g.leader_request("GET", "/v1/collections/docs")[0] == 404


def test_router_survives_leader_failover_in_one_group(sharded):
    url, groups = sharded
    rng = random.Random(1)
    assert call(url, "POST", "/v1/collections", {"name": "c", "dim": DIM})[0] == 201
    body = {"vectors": [{"id": f"a{i}", "embedding": vec(rng)} for i in range(40)]}
    assert call(url, "POST", "/v1/collections/c/upsert", body, timeout=20)[0] == 201
    g = groups[1]
    old = g.leader()
    g.kill(old)
    body = {"vectors": [{"id": f"b{i}", "embedding": vec(rng)} for i in range(40)]}
    st, j = call(url, "POST", "/v1/collections/c/upsert", body, timeout=30)
    assert st == 201, j       # router follows the new leader of group 1
    st, j = call(url, "POST", "/v1/collections/c/count", {}, timeout=30)
    assert st == 200 and j["data"]["count"] == 80
    g.start(old)
