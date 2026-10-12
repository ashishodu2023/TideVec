"""
Experimental multi-process Raft cluster: three tidevec-server processes
replicating writes over HTTP. Covers leader election, follower redirect,
replication of every write type, leader failover and catch-up of a
restarted node from its persisted Raft log.
"""
import json
import random
import subprocess
import time
import urllib.error
import urllib.request

import pytest

from tidevec import NotLeaderError, TideVec

from conftest import _free_ports, _server_binary

SECRET = "test-cluster-secret"
DIM = 8


class Cluster:
    def __init__(self, binary, base_dir, n=3):
        self.binary = binary
        self.ports = _free_ports(n)
        self.urls = [f"http://127.0.0.1:{p}" for p in self.ports]
        self.dirs = [base_dir / f"node{i}" for i in range(n)]
        self.procs = [None] * n

    def start(self, i):
        peers = ",".join(f"{j}={u}" for j, u in enumerate(self.urls))
        self.dirs[i].mkdir(parents=True, exist_ok=True)
        log = open(self.dirs[i].parent / f"node{i}.log", "ab")
        self.procs[i] = subprocess.Popen(
            [self.binary, "--port", str(self.ports[i]), "--data-dir", str(self.dirs[i]),
             "--no-require-auth", "--quiet", "--rate-limit", "100000",
             "--node-id", str(i), "--raft-peers", peers, "--cluster-secret", SECRET],
            stdout=log, stderr=log)
        log.close()
        c = self.client(i)
        try:
            _wait(lambda: c.ping(), what=f"node {i} to start")
        except AssertionError as e:
            tail = (self.dirs[i].parent / f"node{i}.log").read_text(errors="replace")[-2000:]
            raise AssertionError(f"{e}\n--- node{i}.log ---\n{tail}") from None

    def kill(self, i):
        self.procs[i].kill()
        self.procs[i].wait(timeout=10)
        self.procs[i] = None

    def stop_all(self):
        for p in self.procs:
            if p is not None:
                p.terminate()
        for p in self.procs:
            if p is not None:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()

    def client(self, i):
        return TideVec(self.urls[i].removeprefix("http://"), timeout=15)

    def alive(self):
        return [i for i, p in enumerate(self.procs) if p is not None]

    def leader(self):
        for i in self.alive():
            try:
                s = self.client(i).cluster_status()
            except Exception:
                continue
            if s.get("role") == "LEADER":
                return i, s["term"]
        return None


def _wait(pred, timeout=15.0, what="condition"):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            last = pred()
            if last:
                return last
        except Exception as e:  # node may still be starting
            last = e
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what} (last={last!r})")


def _vec(rng):
    return [rng.uniform(-1, 1) for _ in range(DIM)]


@pytest.fixture
def cluster(tmp_path):
    binary = _server_binary()
    if not binary:
        pytest.skip("tidevec-server binary not found")
    c = Cluster(binary, tmp_path)
    for i in range(3):
        c.start(i)
    yield c
    c.stop_all()


def _counts_equal(cluster, name, expected, nodes=None):
    nodes = cluster.alive() if nodes is None else nodes
    def check():
        return all(cluster.client(i).count(name) == expected for i in nodes)
    _wait(check, what=f"{name} to have {expected} vectors on nodes {nodes}")


def test_replication_failover_and_catch_up(cluster):
    rng = random.Random(7)
    leader, term = _wait(cluster.leader, what="a leader")
    follower = next(i for i in range(3) if i != leader)
    _wait(lambda: cluster.client(follower).cluster_status().get("leader") == cluster.urls[leader],
          what="follower to learn the leader")

    # A raw write to a follower is rejected with a leader hint.
    req = urllib.request.Request(
        cluster.urls[follower] + "/v1/collections",
        data=json.dumps({"name": "x", "dim": DIM}).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req, timeout=5)
    assert ei.value.code == 421
    assert json.loads(ei.value.read())["leader"] == cluster.urls[leader]

    # The SDK follows the hint transparently.
    db = cluster.client(follower)
    db.create_collection("docs", dim=DIM, n_shards=2)
    vecs = [{"id": f"v{i}", "embedding": _vec(rng),
             "payload": {"grp": "a" if i % 2 else "b"}} for i in range(40)]
    assert db.upsert("docs", vecs) == 40
    _counts_equal(cluster, "docs", 40)

    assert db.delete("docs", filter="grp=b") == 20
    assert db.delete("docs", ids=["v1", "v3", "missing"]) == 2
    _counts_equal(cluster, "docs", 18)

    # Replicas apply the same write sequence with deterministic index
    # construction, so they answer queries identically.
    q = _vec(rng)
    results = [[h.id for h in cluster.client(i).search("docs", q, top_k=5, temporal_blend=0.0)]
               for i in range(3)]
    assert results[0] == results[1] == results[2]

    # Kill the leader: the survivors elect a new one and keep accepting writes.
    cluster.kill(leader)
    survivor = next(i for i in cluster.alive())
    new_leader, new_term = _wait(
        lambda: (lambda l: l if l and l[0] != leader else None)(cluster.leader()),
        what="a new leader")
    assert new_term > term
    db2 = cluster.client(survivor)
    db2.upsert("docs", [{"id": f"w{i}", "embedding": _vec(rng)} for i in range(10)])
    db2.create_collection("late", dim=DIM, n_shards=1)
    db2.upsert("late", [{"id": "only", "embedding": _vec(rng)}])
    _counts_equal(cluster, "docs", 28)

    # The old leader restarts from its persisted log, rejoins as a
    # follower and catches up on everything it missed.
    cluster.start(leader)
    _counts_equal(cluster, "docs", 28, nodes=[leader])
    _wait(lambda: cluster.client(leader).count("late") == 1, what="late collection on restarted node")
    s = cluster.client(leader).cluster_status()
    assert s["role"] == "FOLLOWER" and s["leader"] == cluster.urls[new_leader]

    # Writes through the restarted node still work (redirected).
    cluster.client(leader).drop_collection("late")
    _wait(lambda: all("late" not in [c.name for c in cluster.client(i).list_collections()]
                      for i in range(3)), what="drop to replicate")


def test_restart_does_not_reapply_log(cluster):
    rng = random.Random(1)
    _wait(cluster.leader, what="a leader")
    db = cluster.client(0)
    db.create_collection("once", dim=DIM, n_shards=1)
    db.upsert("once", [{"id": f"v{i}", "embedding": _vec(rng)} for i in range(5)])
    db.delete("once", ids=["v0"])
    _counts_equal(cluster, "once", 4)

    # Restart every node; replaying the log must not resurrect v0 or fail.
    for i in range(3):
        cluster.kill(i)
    for i in range(3):
        cluster.start(i)
    _wait(cluster.leader, what="a leader after full restart")
    _counts_equal(cluster, "once", 4)
    statuses = [cluster.client(i).cluster_status() for i in range(3)]
    assert all(s["last_applied"] >= 3 for s in statuses)


def test_raft_endpoints_require_cluster_secret(cluster):
    req = urllib.request.Request(
        cluster.urls[0] + "/raft/request_vote",
        data=json.dumps({"term": 999, "candidate_id": 1, "last_log_index": 999,
                         "last_log_term": 999}).encode(),
        method="POST", headers={"Content-Type": "application/json",
                                "X-TideVec-Cluster-Secret": "wrong"})
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(req, timeout=5)
    assert ei.value.code == 403


def test_standalone_reports_mode(server_url):
    assert TideVec(server_url).cluster_status() == {"mode": "standalone"}


def test_not_leader_error_carries_hint():
    e = NotLeaderError("nope", 421, "http://h:1")
    assert e.leader == "http://h:1" and e.status == 421
