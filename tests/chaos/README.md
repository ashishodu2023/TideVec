# TideVec Raft chaos harness (experimental cluster mode)

Black-box fault testing for `tidevec-server --raft-peers ...`, in the
spirit of Jepsen: start a real multi-process cluster, run concurrent
clients, inject faults, record every operation, then check the history
for linearizability and the replicas for divergence.

| File | Purpose |
|---|---|
| `tvcluster.py` | Starts/kills/pauses/wipes local server processes; fault, membership and status helpers |
| `linearizability.py` | Register-model checker (WGL search + memoisation, per key) |
| `test_checker.py` | Unit tests for the checker (known-good and known-bad histories) |
| `chaos_runner.py` | The workload + nemesis + checks; CI and nightly modes |

Related multi-process tests live in `tests/cluster/` (`test_cluster_ops.py`,
`test_multi_raft.py`); the in-process Raft tests are
`tests/cluster/test_raft_cluster.cpp` (`tidevec_cluster_tests`).

## Running

```bash
cmake -S . -B build-raft -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-raft --target tidevec-server tidevec-router tidevec_cluster_tests
export TIDEVEC_SERVER_BIN=$PWD/build-raft/tidevec-server

python -m pytest tests/chaos/test_checker.py          # checker self-tests (no server)
python tests/chaos/chaos_runner.py --mode ci          # ~3 min, 3 nodes, all faults
python tests/chaos/chaos_runner.py --mode nightly     # 1 h, collection epochs every 10 min
python tests/chaos/chaos_runner.py --mode nightly --duration 86400   # 24 h soak
```

Useful options: `--nodes 5`, `--faults kill,partition-leader,...`,
`--read-consistency linearizable|lease|local`, `--seed N` (fault schedule
and client choices; timing still varies), `--out DIR`.
Exit status: 0 pass, 1 linearizability violation or replica divergence,
2 harness error. The output directory holds `chaos.log`,
`report.json`, `history_e<epoch>.jsonl` and every node's data dir and log.

No Python dependencies beyond the standard library (pytest only for the
test files). Only Linux/macOS (uses SIGKILL/SIGSTOP).

## Workload

A set of registers (`--keys`, default 6–8). Each key is a vector id in a
small collection; a write upserts it with payload `{"v": <unique value>}`,
a read is `GET /v1/collections/{c}/vectors/{key}?consistency=linearizable`
(a 404 "Vector not found" reads as the initial empty value). Clients talk
to random nodes and follow 421 leader hints.

Write outcomes:

| Response | Recorded as |
|---|---|
| 200/201 | ok (acknowledged) |
| 421, 404, 429, `503 Write not accepted ...`, connection refused | fail (definitely not applied) |
| `503 Write not confirmed ...`, timeout, connection reset | info (may or may not have been applied) |

`info` writes are allowed to take effect at any point after they were
invoked (or never), as in Knossos/Porcupine.

## Faults (nemesis)

`kill` / `kill-leader` (SIGKILL), `restart-all` (crash the whole cluster),
`pause` (SIGSTOP/SIGCONT, often the leader), `partition-leader`,
`partition-random` (majority/minority), `one-way` (asymmetric link loss),
`lossy-net` (random drops + delay on every node), `slow-disk` (fsync
delay), `disk-full` (every persist fails), `clock-skew` (election/heartbeat
timers run 0.3x–3x and wall clock offset by up to ±5 min), and
`replace-node` (remove a follower, wipe it, join a fresh node id with
`--raft-join` and `add_node`). A heal step after every fault clears all
fault state and restarts dead nodes. Snapshots are taken every 200
entries, so restarts and new members exercise InstallSnapshot.

Network and disk faults use the server's test-only fault hooks:
`POST /v1/admin/fault` exists only with `--enable-fault-injection` (404
otherwise). Body fields: `partition` (ids to cut both ways), `drop_to`,
`drop_from`, `drop_prob`, `rpc_delay_ms`, `fsync_delay_ms`, `disk_full`,
`timer_scale`, `clock_offset_ms`, `clear`. Partitions drop Raft RPCs
(client traffic is unaffected, so clients can still reach a minority
leader — exactly the case linearizable reads must handle).

## Checks

1. **Linearizability** of every key's history (`linearizability.py`):
   Wing–Gong–Lowe search with memoisation of (linearized set, state),
   P-compositionality per key, a step budget (budget exhaustion is
   reported as "unknown", never as a pass or fail).
2. **Convergence / data loss**: after the run every fault is healed,
   all members must apply up to the leader's commit index, a final
   linearizable read of every key is added to the history, and each
   replica's local value must equal the leader's.
3. **Availability** (reported, not asserted): longest stretch without
   a successful client operation and leaderless windows over 1 s.

A `--read-consistency local` run is a checker self-test: follower and
deposed-leader reads are allowed to be stale, so a partition-heavy run
is expected to FAIL (it does: see results below).

## Results so far (development machine, heavily loaded)

| Run | Ops | Faults | Violations |
|---|---|---|---|
| CI mode, 7 seeds (3 nodes, all faults) | ~6–12k each | 18–27 each | 0 |
| CI, 5 nodes | see `report.json` | all | 0 |
| Partitions + 9 node replacements (every original node replaced) | 4.3k | 27 | 0 |
| Lease reads | 13.7k | 18 | 0 |
| Nightly mode with 3 epochs (epoch rotation + per-epoch checks) | 7.2k | 11 | 0 |
| `local` reads + partitions (expected to fail) | 2.2k | 8 | 1 stale read caught |

Bugs found by the harness and fixed:

- **Candidate livelock**: a candidate whose election timed out re-entered
  pre-vote, but pre-votes are only sent by followers, so after a split
  vote no node retried until an outside event (seen as 2–5 s leaderless
  windows after a disk-full leader stepped down). Fixed in `raft.hpp`;
  regression test `candidate_retries_after_failed_election`.
- **Disk-full leader kept leadership** and rejected every write until the
  disk recovered. A leader whose appends fail for longer than the minimum
  election timeout now steps down (it cannot win again until it can
  persist a term). Test `leader_with_persistently_full_disk_steps_down`.
- **404 bodies were overwritten** with "Endpoint not found" by the server's
  catch-all error handler, so "Vector not found"/"Collection not found"
  never reached clients (also broke the register workload's empty reads).
- **Term race in the write path**: the term used to confirm a write was
  read before `submit()`; an election in between could make a different
  entry at the same index look like ours. The write is now confirmed only
  if the entry at that index carries the expected term.

## CI wiring

- Every push and PR (`cpp-test` in `.github/workflows/ci.yml`): `tidevec_cluster_tests`.
- Nightly (`chaos` job, also on manual dispatch): `pytest tests/chaos/test_checker.py`,
  `pytest tests/cluster` (fails on skips) and `chaos_runner.py --mode ci` with the run number as
  seed; the output directory is uploaded as an artifact on failure.
- Not wired yet: `--mode nightly`, `--nodes 5` and `--read-consistency lease` runs, and a
  weekly 24 h soak.

## Known limitations

- Single machine: processes share one disk and clock, so "clock skew" is
  simulated (timer rate + wall-clock offset) and partitions only cover
  Raft RPCs, not client links.
- The register workload covers point reads/writes. Search results,
  graph edges and drift migration are not checked for linearizability.
- Drift migration (`/drift/start`) is not replicated in cluster mode yet
  (returns 501).
