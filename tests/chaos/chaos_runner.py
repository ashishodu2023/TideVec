#!/usr/bin/env python3
"""
TideVec Raft chaos runner (EXPERIMENTAL cluster mode).

Starts a local N-node tidevec-server cluster with --enable-fault-injection,
runs concurrent clients against a register workload while a nemesis
injects faults, records every operation, and checks the history for
linearizability (see linearizability.py) plus final replica convergence.

Workload: each key is a vector id in a small collection; a write upserts
it with payload {"v": <unique value>}, a read is
GET /v1/collections/{c}/vectors/{key}?consistency=linearizable.

Faults: kill -9 (random node or leader), full-cluster crash, SIGSTOP pauses,
leader isolation, majority/minority partitions, one-way link loss, lossy
and slow links, slow fsync, disk full, timer skew + wall-clock offset, and
node replacement (remove a follower, wipe it, join a fresh node). Raft
snapshots are frequent (--snapshot-threshold) so restarts and new nodes
exercise InstallSnapshot.

  python tests/chaos/chaos_runner.py --mode ci        # ~3 minutes
  python tests/chaos/chaos_runner.py --mode nightly   # 1 hour (use --duration 86400 for 24h)

Exit status: 0 = no violations, 1 = linearizability violation or replica
divergence, 2 = harness error (cluster never became available, etc.).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from linearizability import INF, Op, check_history  # noqa: E402
from tvcluster import LocalCluster, call, server_binary  # noqa: E402

EMB = [1.0, 0.0, 0.0, 0.0]

ALL_FAULTS = ["kill", "kill-leader", "restart-all", "pause", "partition-leader",
              "partition-random", "one-way", "lossy-net", "slow-disk", "disk-full",
              "clock-skew", "replace-node"]

MODES = {
    "ci":      dict(duration=150, nodes=3, clients=6, keys=6, epoch=0, quiet=(1.5, 3.0), fault=(2.0, 6.0)),
    "nightly": dict(duration=3600, nodes=3, clients=8, keys=8, epoch=600, quiet=(2.0, 8.0), fault=(3.0, 15.0)),
}


def now() -> float:
    return time.monotonic()


class Runner:
    def __init__(self, args):
        self.args = args
        self.rng = random.Random(args.seed)
        binary = args.binary or server_binary()
        if not binary:
            raise SystemExit("tidevec-server not found (build it or pass --binary / TIDEVEC_SERVER_BIN)")
        self.out = Path(args.out or tempfile.mkdtemp(prefix="tidevec-chaos-"))
        self.out.mkdir(parents=True, exist_ok=True)
        self.cluster = LocalCluster(binary, self.out / "cluster", n=args.nodes,
                                    snapshot_threshold=args.snapshot_threshold,
                                    election_ms=args.election_ms, heartbeat_ms=args.heartbeat_ms)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.epoch = 0
        self.epoch_ops: dict[int, list[Op]] = {0: []}
        self.results: list = []
        self.nemesis_log: list = []
        self.counts = {"ok": 0, "info": 0, "fail": 0}
        self.last_ok = now()
        self.max_gap = 0.0
        self.t0 = now()
        self.errors: list[str] = []

    # ---- logging --------------------------------------------------
    def log(self, msg: str):
        line = f"[{now() - self.t0:8.2f}s] {msg}"
        print(line, flush=True)
        with open(self.out / "chaos.log", "a") as f:
            f.write(line + "\n")

    # ---- collections / epochs ------------------------------------------
    def col(self, epoch: int) -> str:
        return f"reg_e{epoch}"

    def create_collection(self, epoch: int, timeout: float = 120.0):
        deadline = now() + timeout
        body = {"name": self.col(epoch), "dim": len(EMB), "n_shards": 2}
        while now() < deadline:
            st, j = self.cluster.leader_request("POST", "/v1/collections", body, attempts=5)
            if st == 201 or (st in (400, 409) and j and "already exists" in str(j.get("error", ""))):
                return
            time.sleep(0.5)
        raise RuntimeError(f"could not create {self.col(epoch)}")

    # ---- clients ---------------------------------------------------
    def record(self, epoch: int, op: Op):
        with self.lock:
            self.epoch_ops.setdefault(epoch, []).append(op)
            self.counts[op.status] += 1
            if op.status == "ok":
                t = op.ret
                self.max_gap = max(self.max_gap, t - self.last_ok)
                self.last_ok = max(self.last_ok, t)

    def client_loop(self, cid: int):
        rng = random.Random(self.args.seed * 1000 + cid)
        target = rng.choice(list(self.cluster.nodes))
        n = 0
        timeout = self.args.op_timeout
        while not self.stop.is_set():
            epoch = self.epoch
            col = self.col(epoch)
            k = rng.randrange(self.args.keys)
            key = f"{col}/k{k}"
            node = self.cluster.nodes.get(target)
            if node is None or node.removed:
                target = rng.choice([x.id for x in self.cluster.members()])
                continue
            url = node.url
            if rng.random() < 0.5:
                n += 1
                value = f"c{cid}-{n}"
                body = {"vectors": [{"id": f"k{k}", "embedding": EMB, "payload": {"v": value}}]}
                t1 = now()
                st, j = call(url, "POST", f"/v1/collections/{col}/upsert", body, timeout=timeout)
                t2 = now()
                msg = str(j.get("error", "")) if isinstance(j, dict) else ""
                if st in (200, 201):
                    status = "ok"
                elif st is None:
                    status = "info"           # timeout / connection lost: unknown
                elif st == 503 and msg.startswith("Write not confirmed"):
                    status = "info"
                else:
                    status = "fail"           # 421 redirect, refused before append, 404, 429
                    if st not in (0, 421, 503, 404, 429):
                        self.errors.append(f"write {key}: HTTP {st} {msg}")
                meta = {"node": target, "http": st}
                if msg:
                    meta["error"] = msg[:80]
                self.record(epoch, Op(cid, "write", key, value, t1, t2 if status != "info" else INF,
                                      status, meta))
            else:
                path = (f"/v1/collections/{col}/vectors/k{k}?with_vector=false"
                        f"&consistency={self.args.read_consistency}")
                t1 = now()
                st, j = call(url, "GET", path, timeout=timeout)
                t2 = now()
                msg = str(j.get("error", "")) if isinstance(j, dict) else ""
                if st == 200:
                    self.record(epoch, Op(cid, "read", key, j["data"]["payload"].get("v"), t1, t2,
                                          "ok", {"node": target}))
                elif st == 404 and msg.startswith("Vector not found"):
                    self.record(epoch, Op(cid, "read", key, None, t1, t2, "ok", {"node": target}))
                else:
                    self.record(epoch, Op(cid, "read", key, None, t1, t2, "fail",
                                          {"node": target, "http": st, "error": msg[:80]}))
            # Follow leader hints; on silence try another node.
            if st == 421 and isinstance(j, dict) and j.get("leader"):
                hint = next((x.id for x in self.cluster.members() if x.url == j["leader"]), None)
                target = hint if hint is not None else rng.choice([x.id for x in self.cluster.members()])
            elif st in (None, 0, 421, 503):
                target = rng.choice([x.id for x in self.cluster.members()])
                time.sleep(0.02)

    # ---- nemesis ---------------------------------------------------------
    def heal(self):
        c = self.cluster
        for node in list(c.nodes.values()):
            if node.removed:
                continue
            if node.paused:
                c.resume(node.id)
            if not node.alive:
                try:
                    c.start(node.id)
                except AssertionError as e:
                    self.errors.append(f"restart of node {node.id} failed: {e}")
        c.clear_faults()

    def alive_members(self):
        return [n.id for n in self.cluster.members() if n.alive and not n.paused]

    def nemesis_loop(self):
        c = self.cluster
        faults = self.args.faults
        mode = MODES.get(self.args.mode, MODES["ci"])
        while not self.stop.is_set():
            self.stop.wait(self.rng.uniform(*mode["quiet"]))
            if self.stop.is_set():
                break
            fault = self.rng.choice(faults)
            dur = self.rng.uniform(*mode["fault"])
            members = [n.id for n in c.members()]
            lid = c.leader()
            desc = fault
            try:
                if fault in ("kill", "kill-leader"):
                    victim = lid if fault == "kill-leader" and lid is not None else self.rng.choice(members)
                    desc = f"kill -9 node {victim}" + (" (leader)" if victim == lid else "")
                    self.log("nemesis: " + desc)
                    c.kill(victim)
                elif fault == "restart-all":
                    desc = "kill -9 every node, restart"
                    self.log("nemesis: " + desc)
                    for m in members:
                        c.kill(m)
                    dur = min(dur, 2.0)
                elif fault == "pause":
                    victim = lid if lid is not None and self.rng.random() < 0.5 else self.rng.choice(members)
                    desc = f"SIGSTOP node {victim}" + (" (leader)" if victim == lid else "")
                    self.log("nemesis: " + desc)
                    c.pause(victim)
                elif fault == "partition-leader" and lid is not None:
                    desc = f"isolate leader {lid}"
                    self.log("nemesis: " + desc)
                    c.partition([[lid], [m for m in members if m != lid]])
                elif fault in ("partition-random", "partition-leader"):
                    ids = members[:]
                    self.rng.shuffle(ids)
                    cut = self.rng.randint(1, len(ids) // 2)
                    groups = [ids[:cut], ids[cut:]]
                    desc = f"partition {groups}"
                    self.log("nemesis: " + desc)
                    c.partition(groups)
                elif fault == "one-way":
                    a, b = self.rng.sample(members, 2)
                    desc = f"drop messages {a} -> {b}"
                    self.log("nemesis: " + desc)
                    c.fault(a, drop_to=[b])
                    c.fault(b, drop_from=[a])
                elif fault == "lossy-net":
                    p = self.rng.choice([0.1, 0.2, 0.3])
                    d = self.rng.choice([0, 10, 40])
                    desc = f"lossy network drop={p} delay={d}ms"
                    self.log("nemesis: " + desc)
                    for m in members:
                        c.fault(m, drop_prob=p, rpc_delay_ms=d)
                elif fault == "slow-disk":
                    victim = self.rng.choice(members)
                    d = self.rng.choice([50, 150, 400])
                    desc = f"slow fsync {d}ms on node {victim}"
                    self.log("nemesis: " + desc)
                    c.fault(victim, fsync_delay_ms=d)
                elif fault == "disk-full":
                    victim = lid if lid is not None and self.rng.random() < 0.4 else self.rng.choice(members)
                    desc = f"disk full on node {victim}" + (" (leader)" if victim == lid else "")
                    self.log("nemesis: " + desc)
                    c.fault(victim, disk_full=True)
                elif fault == "clock-skew":
                    victim = self.rng.choice(members)
                    scale = self.rng.choice([0.3, 0.5, 2.0, 3.0])
                    off = self.rng.randint(-300_000, 300_000)
                    desc = f"clock skew on node {victim}: timers x{scale}, wall clock {off:+d}ms"
                    self.log("nemesis: " + desc)
                    c.fault(victim, timer_scale=scale, clock_offset_ms=off)
                elif fault == "replace-node":
                    self.replace_node()
                    continue
                self.stop.wait(dur)
            except Exception:  # noqa: BLE001
                self.errors.append(f"nemesis {desc}: {traceback.format_exc(limit=3)}")
            finally:
                self.heal()
                self.nemesis_log.append({"t": round(now() - self.t0, 2), "fault": desc,
                                         "duration": round(dur, 2)})

    def replace_node(self):
        """Remove a follower, wipe it, and join a fresh node in its place."""
        c = self.cluster
        lid = c.leader()
        followers = [m for m in self.alive_members() if m != lid]
        if lid is None or not followers or len(self.alive_members()) < len(c.members()):
            return
        victim = self.rng.choice(followers)
        self.log(f"nemesis: replace node {victim}")
        st, j = self._retry_change(lambda: c.remove_member(victim))
        if st != 200:
            self.log(f"  remove_node {victim} -> {st} {j}")
            return
        c.kill(victim)
        c.wipe(victim)
        new = c.add_slot()
        c.start(new.id, join=True)
        st, j = self._retry_change(lambda: c.add_member(new.id, timeout_ms=60000))
        self.log(f"  add_node {new.id} -> {st}" + ("" if st == 200 else f" {j}"))
        if st != 200:
            # Leave no half-joined node behind; the cluster stays one smaller.
            c.kill(new.id)
            new.removed = True
        self.nemesis_log.append({"t": round(now() - self.t0, 2),
                                 "fault": f"replace node {victim} with {new.id}", "ok": st == 200})

    @staticmethod
    def _retry_change(fn, attempts: int = 30):
        """Membership calls are idempotent; 409 usually means 'previous change still pending'."""
        st, j = None, None
        for _ in range(attempts):
            st, j = fn()
            if st == 200:
                break
            time.sleep(0.5)
        return st, j

    def monitor_loop(self):
        """Logs leader changes and leaderless windows (availability, not safety)."""
        c = self.cluster
        cur, since = None, now()
        self.leaderless = []
        while not self.stop.wait(0.2):
            views = {}
            for m in c.members():
                s = c.status(m.id)
                if s:
                    views[m.id] = (s.get("role"), s.get("term"))
            leaders = sorted((t, i) for i, (r, t) in views.items() if r == "LEADER")
            state = leaders[-1] if leaders else None
            if state != cur:
                t = now()
                if cur is None and t - since > 1.0:
                    self.leaderless.append({"t": round(since - self.t0, 2), "seconds": round(t - since, 2)})
                    self.log(f"monitor: no leader for {t - since:.1f}s")
                self.log(f"monitor: leader {state[1]} term {state[0]}" if state else "monitor: no leader")
                cur, since = state, t

    # ---- checking ----------------------------------------------------------
    def check_epoch(self, epoch: int):
        with self.lock:
            ops = self.epoch_ops.pop(epoch, [])
        with open(self.out / f"history_e{epoch}.jsonl", "w") as f:
            for o in ops:
                f.write(json.dumps({"client": o.client, "kind": o.kind, "key": o.key, "value": o.value,
                                    "call": round(o.call - self.t0, 6),
                                    "ret": None if o.ret == INF else round(o.ret - self.t0, 6),
                                    "status": o.status, "meta": o.meta}, default=str) + "\n")
        t = now()
        res = check_history(ops, budget_per_key=self.args.check_budget)
        bad = [r for r in res if not r.ok]
        unknown = [r for r in res if r.unknown]
        self.results.append({"epoch": epoch, "ops": len(ops), "keys": len(res),
                             "violations": [vars(r) for r in bad],
                             "unknown": [r.key for r in unknown],
                             "check_seconds": round(now() - t, 2)})
        self.log(f"checked epoch {epoch}: {len(ops)} ops, {len(res)} keys, "
                 f"{len(bad)} violation(s), {len(unknown)} unknown, {now() - t:.1f}s")
        for r in bad:
            self.log(f"  VIOLATION {r.key}: {r.reason}")
            for line in r.context:
                self.log(f"    {line}")

    def epoch_loop(self):
        if self.args.epoch <= 0:
            return
        while not self.stop.wait(self.args.epoch):
            old = self.epoch
            try:
                self.create_collection(old + 1)
            except RuntimeError as e:
                self.errors.append(str(e))
                continue
            self.epoch = old + 1
            self.log(f"epoch {old + 1} started")
            time.sleep(self.args.op_timeout * 2 + 1)
            self.check_epoch(old)
            self.cluster.leader_request("DELETE", f"/v1/collections/{self.col(old)}", attempts=10)

    def final_check(self) -> list[str]:
        """Heal, wait for convergence, read every key from every replica."""
        c = self.cluster
        self.heal()
        lid = c.wait_leader(timeout=60)
        problems = []
        epoch, col = self.epoch, self.col(self.epoch)
        st, s = c.request(lid, "GET", "/v1/cluster/status")
        commit = s["data"]["commit_index"] if st == 200 else 0
        try:
            c.wait(lambda: all((c.status(m.id) or {}).get("last_applied", 0) >= commit
                               for m in c.members()), timeout=60, what="replicas to converge")
        except AssertionError as e:
            problems.append(str(e).splitlines()[0])
        finals = {}
        for k in range(self.args.keys):
            key = f"{col}/k{k}"
            t1 = now()
            st, j = c.leader_request("GET", f"/v1/collections/{col}/vectors/k{k}"
                                     "?with_vector=false&consistency=linearizable")
            t2 = now()
            if st == 200:
                finals[k] = j["data"]["payload"].get("v")
            elif st == 404:
                finals[k] = None
            else:
                problems.append(f"final read of {key} failed: HTTP {st}")
                continue
            self.record(epoch, Op(-1, "read", key, finals[k], t1, t2, "ok", {"final": True}))
        for m in c.members():
            for k, want in finals.items():
                st, j = c.request(m.id, "GET", f"/v1/collections/{col}/vectors/k{k}?with_vector=false")
                got = j["data"]["payload"].get("v") if st == 200 else None if st == 404 else f"HTTP {st}"
                if got != want:
                    problems.append(f"replica divergence: node {m.id} has {col}/k{k}={got!r}, "
                                    f"leader has {want!r}")
        return problems

    # ---- main ------------------------------------------------------------
    def run(self) -> int:
        a = self.args
        self.log(f"output: {self.out}")
        self.log(f"mode={a.mode} nodes={a.nodes} duration={a.duration}s clients={a.clients} "
                 f"keys={a.keys} faults={','.join(a.faults)} seed={a.seed} "
                 f"read_consistency={a.read_consistency}")
        c = self.cluster
        try:
            for i in c.initial_ids:
                c.start(i)
            c.wait_leader(timeout=30)
            self.create_collection(0)
        except Exception as e:  # noqa: BLE001
            self.log(f"harness error: cluster did not come up: {e}")
            c.stop_all()
            return 2
        threads = [threading.Thread(target=self.client_loop, args=(i,), daemon=True)
                   for i in range(a.clients)]
        threads.append(threading.Thread(target=self.nemesis_loop, daemon=True))
        threads.append(threading.Thread(target=self.epoch_loop, daemon=True))
        threads.append(threading.Thread(target=self.monitor_loop, daemon=True))
        self.t0 = now()
        self.last_ok = now()
        for t in threads:
            t.start()
        try:
            while now() - self.t0 < a.duration:
                time.sleep(min(10, a.duration))
                with self.lock:
                    cnt = dict(self.counts)
                self.log(f"progress: {cnt} leader={c.leader()} "
                         f"members={[m.id for m in c.members()]}")
        except KeyboardInterrupt:
            self.log("interrupted")
        self.stop.set()
        for t in threads:
            t.join(timeout=a.op_timeout + 30)
        problems = []
        try:
            problems = self.final_check()
        except Exception as e:  # noqa: BLE001
            problems.append(f"final check failed: {e}")
        for ep in sorted(list(self.epoch_ops)):
            self.check_epoch(ep)
        c.stop_all()

        violations = sum(len(r["violations"]) for r in self.results)
        report = {
            "mode": a.mode, "seed": a.seed, "nodes": a.nodes, "duration_s": a.duration,
            "read_consistency": a.read_consistency,
            "ops": self.counts, "max_unavailable_s": round(self.max_gap, 2),
            "leaderless_windows_over_1s": getattr(self, "leaderless", []),
            "violations": violations, "problems": problems,
            "harness_errors": self.errors[:50], "epochs": self.results,
            "nemesis": self.nemesis_log,
        }
        (self.out / "report.json").write_text(json.dumps(report, indent=2, default=str))
        self.log(f"ops: {self.counts}; longest stretch without a successful op: {self.max_gap:.1f}s")
        self.log(f"faults injected: {len(self.nemesis_log)}")
        for p in problems:
            self.log("PROBLEM: " + p)
        if self.errors:
            self.log(f"{len(self.errors)} unexpected client/nemesis errors (see report.json)")
        verdict = "PASS" if violations == 0 and not problems else "FAIL"
        self.log(f"{verdict}: {violations} linearizability violation(s), {len(problems)} other problem(s); "
                 f"report: {self.out / 'report.json'}")
        return 0 if verdict == "PASS" else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=list(MODES) + ["custom"], default="ci")
    p.add_argument("--duration", type=float)
    p.add_argument("--nodes", type=int)
    p.add_argument("--clients", type=int)
    p.add_argument("--keys", type=int)
    p.add_argument("--epoch", type=float, help="seconds per collection epoch (0 = one epoch)")
    p.add_argument("--faults", default=",".join(ALL_FAULTS),
                   help="comma-separated subset of: " + ",".join(ALL_FAULTS))
    p.add_argument("--read-consistency", default="linearizable",
                   choices=["linearizable", "lease", "local"],
                   help="'local' reads are expected to FAIL the check (checker self-test)")
    p.add_argument("--snapshot-threshold", type=int, default=200)
    p.add_argument("--election-ms", type=int, default=300)
    p.add_argument("--heartbeat-ms", type=int, default=75)
    p.add_argument("--op-timeout", type=float, default=3.0)
    p.add_argument("--check-budget", type=int, default=5_000_000)
    p.add_argument("--seed", type=int, default=int(time.time()) % 100000)
    p.add_argument("--binary")
    p.add_argument("--out")
    args = p.parse_args(argv)
    defaults = MODES.get(args.mode, MODES["ci"])
    for k in ("duration", "nodes", "clients", "keys", "epoch"):
        if getattr(args, k) is None:
            setattr(args, k, defaults[k])
    args.faults = [f for f in args.faults.split(",") if f]
    unknown = set(args.faults) - set(ALL_FAULTS)
    if unknown:
        p.error(f"unknown faults: {unknown}")
    return Runner(args).run()


if __name__ == "__main__":
    sys.exit(main())
