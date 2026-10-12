# Backups and restore

## Snapshots

With `--backup` (`TIDEVEC_BACKUP=1` in Docker) the server snapshots its data directory every 6 hours
(`--backup-interval-hours`, `TIDEVEC_BACKUP_INTERVAL_HOURS`) and keeps the newest 48 snapshots. A snapshot is a `tar.gz` of the whole data directory named
`tidevec_<unix-seconds>.tar.gz`, written to `--backup-dir` (`/backups` in the Docker image,
`./backups` for the binary). Each snapshot is appended to `manifests.jsonl` in the same directory.

!!! note "Keep the backup directory outside the data directory"
    Snapshots archive the contents of the data directory. A backup directory inside it is skipped,
    but it then shares the data volume, so losing the volume loses the backups too. Images before
    0.3.0 defaulted to `/data/backups`; move existing snapshots to the new `/backups` volume.

Trigger and inspect snapshots over the API. The `/v1/admin/*` endpoints need the admin API key;
[tenant keys](auth.md#multi-tenant-mode) get `403`.

```bash
curl -X POST -H "X-Api-Key: $TIDEVEC_API_KEY" localhost:6399/v1/admin/backup
curl -H "X-Api-Key: $TIDEVEC_API_KEY" localhost:6399/v1/admin/backups
curl -H "X-Api-Key: $TIDEVEC_API_KEY" localhost:6399/v1/admin/backups/manifests
```

or from Python: `db.trigger_backup()`, `db.list_backups()`, `db.list_backup_manifests()`.

Snapshots are taken while the server keeps running, without pausing writes. The WAL's per-record
checksums let a restored copy discard a torn last record, but writes that arrive during the snapshot
may or may not be included.

## Off-site copies

`--backup-s3 s3://bucket/prefix/` and `--backup-gcs gs://bucket/prefix/` upload each snapshot after
it is written, by running `aws s3 cp` or `gsutil cp`. Those CLIs and their credentials must be
available to the server process. **The Docker image does not include them**, so either mount or
install them in a derived image, or copy `/backups` with a separate job. Upload failures are not
reported by the API, only logged to stderr; check the bucket.

## Restore

```bash
curl -X POST -H "X-Api-Key: $TIDEVEC_API_KEY" -H 'Content-Type: application/json' \
     -d '{"snapshot": "tidevec_1791755538.tar.gz", "confirm": true}' \
     localhost:6399/v1/admin/restore
```

Snapshot names must be `[A-Za-z0-9_.-]+.tar.gz` as listed by `/v1/admin/backups` (`400`
otherwise, `404` if the file does not exist). The server first takes a safety snapshot, extracts the
snapshot into a staging directory inside the data directory, then swaps the data directory's
contents and reloads every collection in place, without a restart (the response has
`restart_required: false`). The data directory itself is not renamed, so this works when it is a
volume mount point, as `/data` is in Docker and Kubernetes. If the swap fails, the previous contents
are put back. `tenants.json` is kept as it is. A record of the restore is written to
`last_restore.json` in the backup directory.

Requests that are already running when the restore starts finish first, against the old data (their
writes are discarded by the restore); new requests wait until the collections are reloaded. In
[cluster mode](#cluster-mode) the endpoint returns `501`.

To restore offline instead, stop the server, empty the data directory and extract the snapshot into
it (`tar -xzf tidevec_<ts>.tar.gz -C <data-dir>`; snapshots taken before 0.3.0 contain a top-level
directory, so add `--strip-components=1`).

## Cluster mode

Snapshots are per node and are not coordinated across a [cluster](cluster.md). Every node holds a
full copy of the collections, but restoring a cluster from a snapshot is not automated or tested.
