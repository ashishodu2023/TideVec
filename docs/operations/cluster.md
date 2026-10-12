# Cluster mode (experimental)

!!! danger "Not production-ready"
    Cluster mode is covered by in-process Raft tests, multi-process tests (replication, failover,
    catch-up, restart, membership changes, auth) and a fault-injection chaos harness with a
    linearizability checker ([`tests/chaos`](https://github.com/ashishodu2023/TideVec/tree/main/tests/chaos)),
    not by production use. Run a single server with backups for anything that matters.

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
- **Reads** are served by whichever node you ask (`consistency=local`, the default), so a follower
  may briefly lag the leader. Search, get/scroll/ids/count and collection stats accept
  `?consistency=linearizable` (the leader confirms its leadership with a quorum first, ReadIndex) or
  `?consistency=lease` (the leader skips that round while its lease holds), also as the
  `X-TideVec-Consistency` header. Followers answer those with `421`.
- Term, vote and log are persisted with `fsync` and survive restarts; a torn log tail is discarded.
  Elections use pre-vote, so a node that was partitioned away does not disrupt the cluster when it
  returns.
- `GET /v1/cluster/status` reports role, term, leader, members and commit/applied indexes.

## Snapshots

Every `--raft-snapshot-threshold` applied entries (default 4096, `0` = never) a node snapshots its
state and truncates the log. A follower too far behind, or a new node, receives the snapshot with
`InstallSnapshot` instead of the full log.

## Adding and removing nodes

```bash
# on the new host: start empty and wait to be added
tidevec-server --port 6399 --data-dir ./data --node-id 3 --raft-peers "$PEERS" --raft-join

# on any node (followers redirect to the leader), with the admin key
curl -X POST -H "X-Api-Key: $TIDEVEC_API_KEY" http://10.0.0.1:6399/v1/admin/cluster/add_node \
     -d '{"id": 3, "url": "http://10.0.0.4:6399"}'
curl -X POST -H "X-Api-Key: $TIDEVEC_API_KEY" http://10.0.0.1:6399/v1/admin/cluster/remove_node \
     -d '{"id": 1}'
```

A new node replicates as a learner until it has caught up, then becomes a voter. Changes are
single-server and one at a time (`409` while another is in progress).

Election and heartbeat timers are tunable with `--raft-election-timeout-ms` (default 400; the
maximum is twice this) and `--raft-heartbeat-ms` (default 100).

## Limitations in 0.3.0

- Every node holds a full copy; there is no sharding across hosts. `tidevec-router` (multi-Raft
  routing by id hash over several clusters) is an experiment, see
  [`DESIGN_MULTI_RAFT.md`](https://github.com/ashishodu2023/TideVec/blob/main/include/tidevec/cluster/DESIGN_MULTI_RAFT.md).
- DriftBridge is disabled (`501`), and `--ultra-durable` is turned off.
- Backups are per node and not coordinated; restore is refused in cluster mode.
- Raft peers talk plain HTTP (the Helm chart refuses `tls.enabled` with `cluster.enabled`).
