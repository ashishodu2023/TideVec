# Cluster mode (experimental)

!!! danger "Not production-ready"
    Cluster mode is covered by multi-process tests (replication, failover, catch-up, restart, auth), not
    by production use. Run a single server with backups for anything that matters.

Three or more `tidevec-server` processes can replicate every write through Raft over HTTP. Every node
holds a **full copy** of all collections: this adds availability and read capacity, not capacity for
more data. There is no sharding across hosts.

## Starting a cluster

```bash
PEERS="0=http://10.0.0.1:6399,1=http://10.0.0.2:6399,2=http://10.0.0.3:6399"
export TIDEVEC_CLUSTER_SECRET=...        # the same on every node; guards the /raft/* endpoints
export TIDEVEC_API_KEY=...               # the same on every node

tidevec-server --port 6399 --data-dir ./data --node-id 0 --raft-peers "$PEERS"   # on 10.0.0.1
tidevec-server --port 6399 --data-dir ./data --node-id 1 --raft-peers "$PEERS"   # on 10.0.0.2
tidevec-server --port 6399 --data-dir ./data --node-id 2 --raft-peers "$PEERS"   # on 10.0.0.3
```

`--raft-peers` lists every node, including the one being started, as `id=url`. In Docker, set
`TIDEVEC_NODE_ID` and `TIDEVEC_RAFT_PEERS` instead. `--raft-verbose` logs elections and replication.

## How it behaves

- **Writes** (create and drop collection, upsert, delete, edges, temporal settings, aliases) go to the
  leader, which appends them to the Raft log and answers once the write is committed and applied
  locally.
- **Followers reject writes** with HTTP `421` and the leader's URL in the `X-TideVec-Leader` header
  and `leader` field. The Python, Go and Java clients follow the hint automatically; Python raises
  `NotLeaderError` if no leader is reachable within its retry budget. A `503` means leadership changed
  before the write was confirmed; retry it.
- **Reads are served locally** by whichever node you ask, so a follower may briefly lag the leader.
- Term, vote and log are persisted with `fsync` and survive restarts; a torn log tail is discarded.
- `GET /v1/cluster/status` reports role, term, leader and commit/applied indexes.

## Limitations in 0.3.0

- No log snapshots or compaction: the Raft log grows without bound.
- No membership changes: the peer list is fixed at startup.
- No linearizable reads.
- DriftBridge is disabled, and `--ultra-durable` is turned off.
- Backups are per node and not coordinated.

!!! info "In progress on `main` (unreleased)"
    Log compaction with snapshot transfer to lagging followers (`--raft-snapshot-threshold`), adding
    and removing nodes (`--raft-join`, `POST /v1/admin/cluster/add_node` and `remove_node`),
    linearizable reads, and tunable election and heartbeat timeouts are being developed. They are not
    part of 0.3.0 and may change.
    <!-- TODO(docs): move these into the sections above once the Phase 3 work is released. -->
