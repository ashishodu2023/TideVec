"""
Local multi-process TideVec Raft cluster driver (stdlib only).

Starts tidevec-server processes on 127.0.0.1, kills / pauses / restarts
them, drives the fault-injection endpoint (/v1/admin/fault, needs
--enable-fault-injection) and the membership admin endpoints. Shared by
the chaos runner and the multi-process cluster tests.
"""
from __future__ import annotations

import http.client
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SECRET = "chaos-cluster-secret"


def server_binary() -> str | None:
    cand = os.environ.get("TIDEVEC_SERVER_BIN")
    if cand and os.path.exists(cand):
        return cand
    for d in ("build-raft", "build", "build-release"):
        p = REPO / d / "tidevec-server"
        if p.exists():
            return str(p)
    return shutil.which("tidevec-server")


def free_ports(n: int) -> list[int]:
    """Distinct free ports below the Linux ephemeral range."""
    rng = random.Random()
    ports: list[int] = []
    while len(ports) < n:
        port = rng.randint(20000, 32000)
        if port in ports:
            continue
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
        ports.append(port)
    return ports


def call(url: str, method: str, path: str, body=None, timeout: float = 5.0,
         headers: dict | None = None):
    """Returns (status, parsed_json_or_None). status None = no response;
    status 0 = connection refused (the request certainly never arrived)."""
    host = url.removeprefix("http://")
    conn = http.client.HTTPConnection(host, timeout=timeout)
    try:
        data = None if body is None else json.dumps(body)
        h = {"Content-Type": "application/json"} if data is not None else {}
        if headers:
            h.update(headers)
        conn.request(method, path, body=data, headers=h)
        resp = conn.getresponse()
        raw = resp.read()
        try:
            return resp.status, json.loads(raw) if raw else None
        except ValueError:
            return resp.status, None
    except ConnectionRefusedError:
        return 0, None
    except (OSError, http.client.HTTPException):
        return None, None
    finally:
        conn.close()


class Node:
    def __init__(self, node_id: int, port: int, data_dir: Path, log_path: Path):
        self.id = node_id
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.dir = data_dir
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.paused = False
        self.removed = False   # no longer a cluster member
        self.joined = False    # added at runtime: restarts use --raft-join

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


class LocalCluster:
    def __init__(self, binary: str, base_dir: Path, n: int = 3, *,
                 snapshot_threshold: int = 4096, election_ms: int = 300,
                 heartbeat_ms: int = 75, fault_injection: bool = True,
                 extra_args: tuple = ()):
        self.binary = binary
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)
        self.snapshot_threshold = snapshot_threshold
        self.election_ms = election_ms
        self.heartbeat_ms = heartbeat_ms
        self.fault_injection = fault_injection
        self.extra_args = tuple(extra_args)
        self.nodes: dict[int, Node] = {}
        self.initial_ids = list(range(n))
        for i, port in zip(self.initial_ids, free_ports(n)):
            self._new_node(i, port)

    # ---- process control --------------------------------------
    def _new_node(self, node_id: int, port: int) -> Node:
        node = Node(node_id, port, self.base / f"node{node_id}", self.base / f"node{node_id}.log")
        self.nodes[node_id] = node
        return node

    def add_slot(self) -> Node:
        """A fresh node id and port (not started, not a member yet)."""
        nid = max(self.nodes) + 1
        return self._new_node(nid, free_ports(1)[0])

    def members(self) -> list[Node]:
        return [n for n in self.nodes.values() if not n.removed]

    def _peers_arg(self, node: Node, join: bool) -> str:
        if join:
            return f"{node.id}={node.url}"
        return ",".join(f"{i}={self.nodes[i].url}" for i in self.initial_ids)

    def start(self, node_id: int, join: bool = False, wait: bool = True):
        node = self.nodes[node_id]
        if node.alive:
            return
        join = join or node.joined
        node.joined = join
        node.dir.mkdir(parents=True, exist_ok=True)
        log = open(node.log_path, "ab")
        args = [self.binary, "--port", str(node.port), "--data-dir", str(node.dir),
                "--no-require-auth", "--quiet", "--rate-limit", "1000000", "--threads", "16",
                "--node-id", str(node.id), "--raft-peers", self._peers_arg(node, join),
                "--cluster-secret", SECRET,
                "--raft-snapshot-threshold", str(self.snapshot_threshold),
                "--raft-election-timeout-ms", str(self.election_ms),
                "--raft-heartbeat-ms", str(self.heartbeat_ms), *self.extra_args]
        if join:
            args.append("--raft-join")
        if self.fault_injection:
            args.append("--enable-fault-injection")
        node.proc = subprocess.Popen(args, stdout=log, stderr=log)
        log.close()
        node.paused = False
        if wait:
            self.wait(lambda: call(node.url, "GET", "/health", timeout=1)[0] == 200,
                      timeout=20, what=f"node {node_id} to start", node=node)

    def kill(self, node_id: int):
        """kill -9"""
        node = self.nodes[node_id]
        if node.proc is None:
            return
        if node.paused:
            node.proc.send_signal(signal.SIGCONT)
        node.proc.kill()
        node.proc.wait(timeout=10)
        node.proc = None
        node.paused = False

    def pause(self, node_id: int):
        node = self.nodes[node_id]
        if node.alive and not node.paused:
            node.proc.send_signal(signal.SIGSTOP)
            node.paused = True

    def resume(self, node_id: int):
        node = self.nodes[node_id]
        if node.alive and node.paused:
            node.proc.send_signal(signal.SIGCONT)
            node.paused = False

    def wipe(self, node_id: int):
        shutil.rmtree(self.nodes[node_id].dir, ignore_errors=True)

    def stop_all(self):
        for n in self.nodes.values():
            if n.proc is not None:
                if n.paused:
                    n.proc.send_signal(signal.SIGCONT)
                n.proc.terminate()
        for n in self.nodes.values():
            if n.proc is not None:
                try:
                    n.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    n.proc.kill()
                n.proc = None

    # ---- queries ------------------------------------------------
    def request(self, node_id: int, method: str, path: str, body=None, timeout: float = 5.0):
        return call(self.nodes[node_id].url, method, path, body, timeout)

    def status(self, node_id: int) -> dict | None:
        node = self.nodes[node_id]
        if not node.alive or node.paused:
            return None
        st, j = call(node.url, "GET", "/v1/cluster/status", timeout=1)
        return j.get("data") if st == 200 and j else None

    def leader(self) -> int | None:
        """The leader with the highest term among reachable nodes."""
        best = None
        for n in self.members():
            s = self.status(n.id)
            if s and s.get("role") == "LEADER" and (best is None or s["term"] > best[1]):
                best = (n.id, s["term"])
        return best[0] if best else None

    def wait_leader(self, timeout: float = 20.0, exclude: int | None = None) -> int:
        return self.wait(lambda: (lambda l: l if l is not None and l != exclude else None)(self.leader()),
                         timeout=timeout, what="a leader")

    def wait(self, pred, timeout: float = 20.0, what: str = "condition", node: Node | None = None):
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            try:
                last = pred()
                if last is not None and last is not False:
                    return last
            except Exception as e:  # noqa: BLE001 - nodes may be starting
                last = e
            if node is not None and node.proc is not None and node.proc.poll() is not None:
                break
            time.sleep(0.05)
        tail = ""
        if node is not None and node.log_path.exists():
            tail = "\n--- " + node.log_path.name + " ---\n" + node.log_path.read_text(errors="replace")[-3000:]
        raise AssertionError(f"timed out waiting for {what} (last={last!r}){tail}")

    def leader_request(self, method: str, path: str, body=None, timeout: float = 10.0,
                       attempts: int = 60):
        """Sends to the current leader, following 421 hints and retrying."""
        last = (None, None)
        for _ in range(attempts):
            lid = self.leader()
            if lid is not None:
                last = self.request(lid, method, path, body, timeout)
                if last[0] not in (None, 0, 421, 503):
                    return last
            time.sleep(0.2)
        return last

    # ---- faults & membership ------------------------------------
    def fault(self, node_id: int, **kw):
        node = self.nodes[node_id]
        if not node.alive or node.paused:
            return None
        return call(node.url, "POST", "/v1/admin/fault", kw, timeout=2)[0]

    def clear_faults(self):
        for n in self.nodes.values():
            self.fault(n.id, clear=True)

    def partition(self, groups: list[list[int]]):
        """Each node can only talk to nodes in its own group."""
        ids = [n.id for n in self.nodes.values()]
        for g in groups:
            others = [i for i in ids if i not in g]
            for i in g:
                self.fault(i, partition=others)

    def add_member(self, node_id: int, timeout_ms: int = 30000):
        node = self.nodes[node_id]
        st, j = self.leader_request("POST", "/v1/admin/cluster/add_node",
                                    {"id": node_id, "url": node.url, "timeout_ms": timeout_ms},
                                    timeout=timeout_ms / 1000 + 5)
        if st == 200:
            node.removed = False
        return st, j

    def remove_member(self, node_id: int, timeout_ms: int = 30000):
        st, j = self.leader_request("POST", "/v1/admin/cluster/remove_node",
                                    {"id": node_id, "timeout_ms": timeout_ms},
                                    timeout=timeout_ms / 1000 + 5)
        if st == 200:
            self.nodes[node_id].removed = True
        return st, j
