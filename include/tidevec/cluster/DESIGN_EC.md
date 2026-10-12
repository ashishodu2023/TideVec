# Cross-host erasure coding — design (not implemented)

Status: design only. Today RS(10,4) (`erasure/reed_solomon.hpp`,
`UltraDurableCollection`) encodes segments with all 14 shards on one
host, so the host is the failure domain and the "8.7 nines" figure is
what the code allows across independent disks, not what a deployment
gets. Multi-process cluster mode (`--raft-peers`) gives 3× (or 5×) full
replication instead. This document describes how to get EC durability
and ~1.4× storage across hosts.

## What gets coded

Only **immutable, sealed segment files** (`storage/segment_store.hpp`,
`.cvec` rows + PQ/index side files). The mutable parts stay replicated:

| Data | Protection |
|---|---|
| Raft log + WAL tail (recent writes) | Raft replication (3 or 5 copies), fsync on a majority before ack |
| Write buffer / unsealed segment | in the replicated log until sealed |
| Sealed segments (bulk of the bytes) | RS(k, m) stripes across hosts |
| Index (HNSW/TV graph), metadata | rebuilt from segments, or coded like segments |

This is the usual "replicate hot, erasure-code cold" split (Ceph EC
pools, HDFS-EC, Azure LRC): the log is small and needs low-latency
quorum writes; segments are large and written once.

## Stripe layout

- A sealed segment is split into `k` equal data chunks (padded), and
  `m` parity chunks are computed: RS(10,4) by default; RS(6,3) for
  clusters with fewer than 14 failure domains.
- Each of the `k+m` chunks goes to a **different failure domain**
  (host, then rack/zone when known). With fewer than `k+m` hosts the
  scheme must degrade explicitly (e.g. RS(4,2) on 6 hosts) — never put
  two chunks of a stripe on one host silently.
- Chunk = `{segment_id, stripe_index, chunk_index, length, crc32c}` +
  bytes; segment manifest = `{segment_id, k, m, size, sha256,
  chunk → host map, placement_version}`.
- The **manifest is replicated state**: committed through the owning
  shard group's Raft log (a `SEAL` entry) only after at least `k + 1`
  chunks are durably stored (fsync acknowledged by each chunk host), so
  a committed manifest always references a decodable stripe even if one
  more host fails during sealing.

## Write path (sealing)

1. Leader of the shard group decides a segment is full; it is still
   fully covered by the replicated log/WAL.
2. Leader encodes the stripe and sends chunks to chunk hosts
   (`PUT /v1/internal/chunks/{segment}/{i}`, authenticated with the
   cluster secret), in parallel.
3. When `k + 1` acks arrive (all `k + m` normally), the leader appends
   `SEAL{manifest}`. On apply, every replica drops its full copy of the
   segment and keeps only what it is assigned (its chunk, if any) plus
   the manifest. WAL/log entries covered by the segment become
   eligible for compaction (this is also what bounds Raft snapshot
   size, which today grows with the WAL).
4. Chunks never acknowledged before the timeout are retried by the
   repair loop below.

## Read path

- Hot path does not decode: each replica keeps an in-memory/SSD cache
  of the segments it serves plus PQ codes; ANN search uses PQ for
  candidates and fetches full rows for re-scoring.
- A row fetch for a coded segment reads the **one data chunk** that
  contains the row (systematic code: data chunks are plain bytes), from
  the host that holds it. Only when that host is down is a degraded
  read done: fetch any `k` chunks of the stripe and decode the range.
- Large scans (rebuilds, snapshots) prefer reconstructing whole
  segments locally.

## Repair

- Each shard-group leader runs a scrubber over its manifests:
  `HEAD` every chunk periodically, full CRC verification at a lower
  rate (bit rot), and on host loss (membership removal or failure
  detector timeout) enumerate stripes with chunks on that host.
- Repair = read any `k` surviving chunks, re-encode the missing ones,
  write them to new hosts chosen by the placement policy, commit an
  updated manifest (`RECHUNK` entry). Prioritise stripes with the
  fewest surviving chunks.
- Repair bandwidth is the cost of RS: one lost chunk needs `k` chunk
  reads. If that becomes the bottleneck, switch to an LRC (local
  parities, e.g. Azure LRC(12,2,2)) — the manifest already carries the
  code parameters.

## Durability model (to replace the single-host claim)

With chunks on independent hosts, per-host annual failure `p_h`, and
repair time `T_r`, data is lost only if `m + 1` chunks of one stripe
are lost within a repair window. With `p_h = 2%/yr`, `T_r = 6 h`,
RS(10,4): ≈ `C(14,5) (p_h T_r/1y)^5` per stripe-window — many more nines
than the no-repair disk model, but this must be computed from measured
host failure rates and real repair times, and reported per deployment.
The recent-writes window is protected by Raft replication only.

## Interaction with Raft and sharding

- EC is per shard group: chunk hosts may be any cluster nodes, not
  just the group's Raft members (spreads load across all hosts).
- Raft snapshots for EC'd data contain manifests, not segment bytes —
  InstallSnapshot stays small; a new member fetches chunks lazily.
- Node replacement: Raft membership change (implemented) + repair of
  that host's chunks (this design).

## Implementation steps

1. Chunk store service on every node (`PUT/GET/HEAD/DELETE` chunks,
   CRC on write and read, fsync).
2. `SEAL` / `RECHUNK` log entries and manifest table in the replicated
   state machine; WAL compaction after `SEAL`.
3. Degraded-read path in `SegmentStore` (row → chunk → host).
4. Scrubber + repair loop with priority queue; metrics: stripes at
   risk, repair backlog, bytes repaired.
5. Chaos: kill/wipe up to `m` chunk hosts during writes and repairs,
   verify every acknowledged vector is readable (extend
   `tests/chaos/chaos_runner.py` with a "verify all acked ids" check).
