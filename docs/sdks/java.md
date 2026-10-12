# Java SDK

Java 17+, no runtime dependencies (uses `java.net.http`).

```xml
<dependency>
  <groupId>io.tidevec</groupId>
  <artifactId>tidevec-java</artifactId>
  <version>0.3.0</version>
</dependency>
```

!!! note "Where to get it"
    Release builds are deployed to GitHub Packages. Publishing to Maven Central is not set up yet;
    until then, add the GitHub Packages repository or build from `sdk/java` with `mvn install`.

```java
import io.tidevec.TideVecClient;
import io.tidevec.TideVecClient.*;

try (TideVecClient db = TideVecClient.builder("localhost", 6399)
        .apiKey(System.getenv("TIDEVEC_API_KEY"))
        .build()) {

    db.createCollection(CollectionConfig.builder("kb")
            .dim(384)
            .halfLifeMs(30L * 24 * 3600 * 1000)
            .temporalBlend(0.3f)
            .supersession(0.5f, "doc_id")
            .build());

    db.upsert("kb", List.of(
            Vector.builder("refunds-2026", embedding)
                  .payload(Map.of("doc_id", "refunds"))
                  .build()));

    SearchResponse r = db.search("kb", query, SearchOptions.builder()
            .topK(5)
            .supersededPenalty(1.0f)
            // .temporalBlend(0f)    // unset = collection setting
            // .filter("lang=en")
            .build());
    for (SearchHit h : r.hits()) {
        System.out.println(h.id() + " " + h.score() + (h.isSuperseded() ? " (outdated)" : ""));
    }

    int removed = db.deleteWhere("kb", "source=crm");
}
```

The builder also takes `timeout(Duration)`, `tls(boolean)` and `maxRetries(int)` (for `429`). The
client follows `421` leader redirects in [cluster mode](../operations/cluster.md).

## Methods

| Area | Methods |
|---|---|
| Collections | `createCollection`, `dropCollection`, `getCollection`, `listCollections`, `count`, `setTemporal` |
| Vectors | `upsert`, `delete(collection, ids, filter)`, `deleteWhere(collection, filter)`, `deleteVectors(collection, ids)`, `addEdges`, `search` |
| Server | `ping`, `info`, `clusterStatus` |
| DriftBridge | `startDrift`, `driftStatus`, `abortDrift` |
| Backups | `triggerBackup`, `listBackups`, `restoreBackup` |

Non-2xx responses throw `TideVecException` (an `IOException`) with `status()`, `leader()` and
`isNotFound()`.

!!! warning "Breaking in 0.3.0"
    `SearchOptions.temporalBlend` is a nullable `Float` (unset means the collection's setting), and
    `deleteVectors` returns the number removed.

Retrieval and the other 0.3.0 APIs: `get`, `scroll`, `listIds`, `count` (with filter / `as_of`),
`setAlias` / `deleteAlias` / `listAliases`, `compact` / `compactionStatus`; `SearchOptions` carries
structured filters, `as_of` and hybrid text queries. See
[Retrieval, filters, as-of and hybrid](../concepts/filters.md).
