"""
Multi-process Raft cluster operations (EXPERIMENTAL cluster mode):
log compaction + InstallSnapshot, node replacement under continuous
writes, linearizable / lease reads, partitions via the fault-injection
endpoint, and the endpoint being absent without --enable-fault-injection.

    TIDEVEC_SERVER_BIN=build-raft/tidevec-server pytest tests/cluster/test_cluster_ops.py
"""
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chaos"))
from tvcluster import LocalCluster, call, server_binary  # noqa: E402

BIN = server_binary()
pytestmark = pytest.mark.skipif(BIN is None, reason="tidevec-server not built")
EMB = [1.0, 0.0, 0.0, 0.0]


@pytest.fixture
def make_cluster(tmp_path):
    clusters = []

    def make(n=3, **kw):
        c = LocalCluster(BIN, tmp_path / f"c{len(clusters)}", n=n, **kw)
        clusters.append(c)
        for i in c.initial_ids:
            c.start(i)
        c.wait_leader(timeout=30)
        return c

    yield make
    for c in clusters:
        c.stop_all()


def create(c, name="col"):
    st, j = c.leader_request("POST", "/v1/collections", {"name": name, "dim": len(EMB)})
    assert st == 201, j


def upsert(c, ids, name="col", node=None):
    body = {"vectors": [{"id": i, "embedding": EMB, "payload": {"v": i}} for i in ids]}
    if node is None:
        return c.leader_request("POST", f"/v1/collections/{name}/upsert", body, timeout=15)
    return c.request(node, "POST", f"/v1/collections/{name}/upsert", body, timeout=15)


def count_local(c, node, name="col"):
    st, j = c.request(node, "POST", f"/v1/collections/{name}/count", {})
    return j["data"]["count"] if st == 200 else None


def converged(c):
    lid = c.leader()
    if lid is None:
        return False
    commit = c.status(lid)["commit_index"]
    return all((c.status(m.id) or {}).get("last_applied", -1) >= commit for m in c.members())


def test_fault_endpoint_is_absent_without_flag(make_cluster):
    c = make_cluster(fault_injection=False)
    lid = c.leader()
    st, _ = c.request(lid, "GET", "/v1/admin/fault")
    assert st == 404
    st, _ = c.request(lid, "POST", "/v1/admin/fault", {"partition": [1]})
    assert st == 404
    assert c.status(lid)["fault_injection"] is False


def test_snapshot_compaction_and_install_snapshot_to_lagging_node(make_cluster):
    c = make_cluster(snapshot_threshold=20)
    create(c)
    lid = c.leader()
    f = next(m.id for m in c.members() if m.id != lid)
    c.kill(f)
    for i in range(100):                    # one log entry per write
        st, j = upsert(c, [f"v{i}"])
        assert st in (200, 201), j
    lid = c.wait_leader()
    s = c.status(lid)
    assert s["snapshot_index"] >= 80, s     # log compacted past the follower
    c.start(f)                              # its log ends before the snapshot
    c.wait(lambda: converged(c), timeout=30, what="follower to catch up")
    assert c.status(f)["snapshot_index"] >= 80
    assert count_local(c, f) == 100
    st, j = c.request(f, "GET", "/v1/collections/col/vectors/v99")
    assert st == 200 and j["data"]["payload"]["v"] == "v99"
    # A wiped node also rebuilds purely from the snapshot.
    c.kill(f)
    c.wipe(f)
    c.start(f)
    c.wait(lambda: converged(c) and count_local(c, f) == 100, timeout=30,
           what="wiped follower to restore from snapshot")


def test_replace_every_node_without_downtime(make_cluster):
    c = make_cluster(snapshot_threshold=50)
    create(c)
    acked, stop, gaps = [], threading.Event(), []

    def writer():
        n, last_ok = 0, time.monotonic()
        while not stop.is_set():
            n += 1
            st, _ = upsert(c, [f"w{n}"])
            t = time.monotonic()
            if st in (200, 201):
                acked.append(f"w{n}")
                gaps.append(t - last_ok)
                last_ok = t
            else:
                time.sleep(0.05)

    t = threading.Thread(target=writer, daemon=True)
    t.start()
    time.sleep(1)
    for old in list(c.initial_ids):          # ends by removing the leader itself
        st, j = c.remove_member(old)
        assert st == 200, j
        c.kill(old)
        new = c.add_slot()
        c.start(new.id, join=True)
        st, j = c.add_member(new.id, timeout_ms=30000)
        assert st == 200, j
        time.sleep(1)
    time.sleep(1)
    stop.set()
    t.join(timeout=30)
    members = [m.id for m in c.members()]
    assert not set(members) & set(c.initial_ids)
    assert len(members) == 3
    assert len(acked) > 50
    assert max(gaps) < 5.0, f"writes stalled for {max(gaps):.1f}s"
    c.wait(lambda: converged(c), timeout=30, what="new members to converge")
    for m in members:
        assert count_local(c, m) == len(set(acked)), m
    lid = c.leader()
    assert sorted(x["id"] for x in c.status(lid)["members"]) == sorted(members)


def test_linearizable_and_lease_reads(make_cluster):
    c = make_cluster()
    create(c)
    assert upsert(c, ["a"])[0] in (200, 201)
    lid = c.leader()
    f = next(m.id for m in c.members() if m.id != lid)
    path = "/v1/collections/col/vectors/a"
    for mode in ("linearizable", "lease"):
        st, j = c.request(lid, "GET", f"{path}?consistency={mode}")
        assert st == 200 and j["data"]["payload"]["v"] == "a"
        st, j = c.request(f, "GET", f"{path}?consistency={mode}")
        assert st == 421 and j["leader"] == c.nodes[lid].url
    st, _ = call(c.nodes[lid].url, "GET", path, headers={"X-TideVec-Consistency": "linearizable"})
    assert st == 200
    st, _ = c.request(lid, "GET", f"{path}?consistency=bogus")
    assert st == 400
    c.wait(lambda: c.request(f, "GET", path)[0] == 200, timeout=10, what="local follower read")
    # Missing vectors are a 404 with a specific message, also under linearizable reads.
    st, j = c.request(lid, "GET", "/v1/collections/col/vectors/nope?consistency=linearizable")
    assert st == 404 and "Vector not found" in j["error"]
    # An isolated leader must not serve linearizable reads.
    others = [m.id for m in c.members() if m.id != lid]
    c.partition([[lid], others])
    st, _ = c.request(lid, "GET", f"{path}?consistency=linearizable", timeout=10)
    assert st in (421, 503)
    c.clear_faults()


def test_minority_partition_cannot_commit_and_heals(make_cluster):
    c = make_cluster()
    create(c)
    old = c.leader()
    others = [m.id for m in c.members() if m.id != old]
    c.partition([[old], others])
    st, j = upsert(c, ["unacked"], node=old)
    assert st in (421, 503), (st, j)        # never acknowledged without a quorum
    new = c.wait(lambda: next((i for i in others
                               if (c.status(i) or {}).get("role") == "LEADER"), None),
                 timeout=15, what="majority to elect a leader")
    st, j = upsert(c, ["after"], node=new)
    assert st in (200, 201), j
    c.clear_faults()
    c.wait(lambda: converged(c) and count_local(c, old) == count_local(c, new), timeout=20,
           what="old leader to rejoin")
    st, _ = c.request(old, "GET", "/v1/collections/col/vectors/after")
    assert st == 200
