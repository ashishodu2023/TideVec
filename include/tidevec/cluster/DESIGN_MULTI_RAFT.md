# Multi-Raft sharding — design (EXPERIMENTAL)

Status: step 1 implemented (`multi_raft_router.hpp`, `tidevec-router`,
`tests/cluster/test_multi_raft.py`). Rebalancing and placement changes
are design-only.

## Goal

Scale a TideVec cluster past one Raft group (one leader doing every
write, every node holding every vector) to many hosts, e.g. 100M vectors
over 6+ hosts, without giving up per-key linearizability.

## Model

```
          clients
             │
     tidevec-router (stateless, N instances)
      │         │          │
   group g0   group g1   group g2      each = 3 (or 5) tidevec-server
   [0, a]     (a, b]     (b, 2^64)     processes running Raft
```

- **Shard group** = an ordinary Raft cluster (`--raft-peers`). Every
  collection exists in every group; each group stores the vectors whose
  id hash falls into its range.
- **Hash**: `shard_hash(id) = fmix64(fnv1a64(id))` over 64 bits.
  (Plain FNV-1a put every `doc-N` id in the same half; the murmur3
  finalizer fixes the high-bit spread.) Ranges, not modulo, so a range
  can later be split or moved without rehashing everything.
- **Placement map** (`Placement`): `version`, ordered contiguous ranges
  covering `[0, 2^64-1]`, replica URLs per group. Today it is a static
  JSON file or `--groups` (equal split) given to the router.

## Request routing (implemented)

| Operation | Routing |
|---|---|
| upsert | split by owner, forward each sub-batch to that group's leader in parallel |
| delete by ids | split by owner; delete by filter is broadcast |
| GET vectors/{id} | owner group (consistency param passed through) |
| search | broadcast, merge by `score`, keep `top_k` |
| count / GET collection | broadcast, sum |
| create / drop collection | broadcast; "already exists"/"not found" on some groups is tolerated so retries converge |
| everything else | 404 "not supported by tidevec-router" |

Leader discovery: the router caches each group's leader, follows 421
`leader` hints and walks the replica list on connection errors or 503.
Forwarded writes are idempotent (upsert of the same id/value, delete of
ids), so the router retries "Write not confirmed" responses.

### Consistency

- **Per key**: linearizable, because a key lives in exactly one group and
  that group is a linearizable register (reads with
  `?consistency=linearizable`).
- **Across keys / groups**: no atomicity. A multi-group upsert can
  partially succeed (router returns 503 listing per-group results;
  retrying is safe). A scatter-gather search is not a snapshot across
  groups — each group answers at its own commit point.
- **Graph edges across groups** are not supported: `edges` is not routed
  (an edge is stored with its source vector; causal expansion only sees
  edges within one group). Future: route edges by source id and resolve
  remote targets lazily.
- **Temporal / `as_of` queries** work per group because timestamps are
  normalised by each group's leader when the entry is created.

## Rebalancing (design only)

Moving range `R` from group A to group B, online:

1. **Placement epoch.** The placement map becomes replicated state: a
   small dedicated Raft group (or group g0's log) stores
   `{version, ranges, groups, migrations}`. Routers watch it
   (`GET /v1/cluster/placement?wait_version=v`, long poll) and every
   forwarded request carries `X-TideVec-Placement-Version`; a group
   rejects requests for ranges it does not own at that version with
   `409 wrong shard` so stale routers refresh.
2. **Copy.** Mark `R` as `migrating A→B` (new version). A streams a
   point-in-time copy of `R` to B (scroll over ids in R with `as_of`
   = migration start, B upserts through its own Raft log).
3. **Catch-up.** A keeps serving R; writes to R committed in A after the
   copy point are forwarded to B via a change stream (A's applied log
   entries filtered by range, tailed from the migration start index).
4. **Cut-over.** A briefly blocks writes to R (fence entry in A's log),
   B applies the remaining tail, the placement group commits
   `R → B` (version + 1). A rejects R from then on; routers retry on B.
   Unavailability for R = one fence + one placement commit (~2 RTT).
5. **Cleanup.** A deletes R's vectors in the background.

Splitting a hot range is the same with B = a new group. Merging is the
reverse. Node replacement inside a group does not need any of this —
it is Raft membership (`add_node` / `remove_node`), already implemented.

## Exit-criteria gap: 100M vectors over 6 hosts

Not measured yet. Needed: a load generator through `tidevec-router`
(batch upserts, mixed search), 2 groups × 3 hosts (or 3 groups × 2 with
5-node groups spanning hosts), report write throughput, p50/p99 search
latency vs. single group, and recovery time after killing a host. Main
expected bottlenecks: per-entry fsync in the Raft log (batch commits),
file-level state snapshots (size grows with the WAL until WAL compaction
exists), and scatter-gather fan-out cost for small `top_k`.
