# Durability

## What protects your data on a single server

- **Write-ahead log.** Every change (upsert, delete, edge) is appended to a per-shard WAL with a
  CRC per record and replayed on start. A torn record at the tail (from a crash mid-write) is
  discarded.
- **Segment store** (`--segment-store`, on by default): vectors are also kept in disk-backed
  segments.
- **Ultra-durable stack** (`--ultra-durable`, on by default, and in the Docker image):
    - segments encoded with **Reed-Solomon RS(10,4)** over GF(2⁸): any 10 of the 14 shards
      reconstruct the data;
    - WAL **group commit**: one `fsync` per batch of writes instead of one per write;
    - a **health monitor** that scrubs checksums, tracks lost shards and rebuilds them;
    - an in-process Raft log that orders writes.
- **Backups**: periodic or on-demand snapshots, optionally uploaded to S3 or GCS. See
  [Backups and restore](backups.md).

## Modelled durability

These figures come from `ReedSolomon::durability_nines` and its unit tests, assuming independent disks
that each fail with 0.4% annual probability (the Backblaze annualised failure rate) and **no repair
during the year**:

| Scheme | Storage overhead | Shard losses survived | Modelled durability |
|---|---:|---:|---:|
| 3-way replication | 3.0× | 2 | 7.2 nines |
| RS(6,3) | 1.5× | 3 | 7.5 nines |
| **RS(10,4)** | **1.4×** | **4** | **8.7 nines** |
| RS(14,4) | 1.29× | 4 | 8.1 nines |

!!! warning "A property of the coding scheme, not of a deployment"
    On a standalone server **all 14 shards live on one host**. Losing that host's disk loses the data
    regardless of the coding. Reed-Solomon protects against corruption and partial loss of shard
    files, not against losing the machine. Use backups stored elsewhere.

    The in-process Raft group gives ordering, not availability: all its members run in the same
    process. Repairing lost shards within hours would raise the modelled figures by orders of
    magnitude, but that is not modelled.

Before 0.3.0 the model added two unjustified nines (RS(10,4) was reported as 10.7 nines); the
figures above are the corrected ones (see the [changelog](../changelog.md)).

## Replication across hosts

[Cluster mode](cluster.md) replicates every write to three or more servers through Raft, so a write
acknowledged by the leader survives the loss of a minority of nodes. It is experimental, and
`--ultra-durable` is turned off in that mode.
