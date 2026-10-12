# Go SDK

```bash
go get github.com/ashishodu2023/TideVec/sdk/go
```

Go 1.21+, standard library only. The module lives in a subdirectory, so its versions are tags named
`sdk/go/vX.Y.Z` (see [Releasing](../operations/releasing.md#go-module)).

```go
import (
    "context"
    "os"

    "github.com/ashishodu2023/TideVec/sdk/go/tidevec"
)

ctx := context.Background()
db, err := tidevec.New("localhost:6399", tidevec.ClientConfig{APIKey: os.Getenv("TIDEVEC_API_KEY")})
if err != nil { panic(err) }

err = db.CreateCollection(ctx, tidevec.CollectionConfig{
    Name: "kb", Dim: 384, HalfLifeMs: 30 * 24 * 3600 * 1000, TemporalBlend: 0.3,
    Supersession: &tidevec.SupersessionConfig{SimilarityThreshold: 0.5, MatchField: "doc_id"},
})

err = db.Upsert(ctx, "kb", []tidevec.Vector{
    {ID: "refunds-2026", Embedding: vec, Payload: map[string]string{"doc_id": "refunds"}},
})

resp, err := db.Search(ctx, "kb", query, tidevec.SearchOptions{
    TopK: 5,
    SupersededPenalty: 1.0,
    // TemporalBlend: tidevec.Blend(0),   // nil = collection setting; Blend(0) = pure similarity
    // Filter: "lang=en",
})
for _, h := range resp.Hits {
    fmt.Println(h.ID, h.Score, h.IsSuperseded(), h.SupersededBy)
}

n, err := db.Delete(ctx, "kb", nil, "source=crm")   // ids and/or filter; returns count
```

`ClientConfig` has `APIKey` (falls back to `$TIDEVEC_API_KEY`), `TLS`, `TimeoutMs` and `MaxRetries`
(for `429`). The client is safe for concurrent use and follows `421` leader redirects in
[cluster mode](../operations/cluster.md).

## Methods

| Area | Methods |
|---|---|
| Collections | `CreateCollection`, `DropCollection`, `GetCollection`, `ListCollections`, `Count`, `SetTemporal` |
| Vectors | `Upsert`, `Delete(ids, filter)`, `DeleteVectors(ids)`, `AddEdges`, `Search` |
| Server | `Ping`, `Info`, `ClusterStatus` |
| DriftBridge | `StartDrift`, `DriftStatus`, `AbortDrift` |
| Backups | `TriggerBackup`, `ListBackups`, `ListBackupManifests`, `RestoreBackup` |

Errors are `*tidevec.APIError` (`Status`, `Message`, and `Leader` on `421`); `tidevec.IsNotFound(err)`
checks for `404`.

!!! warning "Breaking in 0.3.0"
    `SearchOptions.TemporalBlend` is now `*float32` (`nil` means the collection's setting, so
    `Blend(0)` is the only way to ask for pure similarity). `Delete` takes `(ids, filter)` and returns
    the number removed.

Retrieval and the other 0.3.0 APIs: `Get`, `GetMany`, `Scroll`, `ListIDs`, `CountWhere`,
`DeleteWhere`, `SetAlias` / `DeleteAlias` / `ListAliases`, `Compact` / `CompactionStatus`;
`SearchOptions` carries structured filters, `as_of` and hybrid text queries. See
[Retrieval, filters, as-of and hybrid](../concepts/filters.md).
