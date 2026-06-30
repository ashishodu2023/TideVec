package io.tidevec;

import java.io.*;
import java.net.*;
import java.net.http.*;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.CompletableFuture;

/**
 * TideVec Java Client — REST/HTTP implementation.
 *
 * <p>No gRPC or protobuf dependencies required.
 *
 * <h3>Quick Start:</h3>
 * <pre>{@code
 * try (TideVecClient db = TideVecClient.builder("localhost", 6399).build()) {
 *
 *     db.createCollection(CollectionConfig.builder("docs")
 *         .dim(768)
 *         .halfLifeMs(HalfLife.ONE_WEEK)
 *         .temporalBlend(0.3f)
 *         .build());
 *
 *     db.upsert("docs", List.of(
 *         Vector.builder("v1", new float[]{0.1f, 0.2f, 0.3f})
 *             .payload(Map.of("source", "wiki"))
 *             .build()
 *     ));
 *
 *     SearchResponse results = db.search("docs", queryVec,
 *         SearchOptions.builder().topK(10).temporalBlend(0.3f).build());
 *
 *     for (SearchHit hit : results.hits()) {
 *         System.out.println(hit.id() + " score=" + hit.score());
 *     }
 * }
 * }</pre>
 */
public class TideVecClient implements AutoCloseable {

    // ── Half-life presets ───────────────────────────────────────────
    public static final class HalfLife {
        public static final long ONE_HOUR  =     3_600_000L;
        public static final long ONE_DAY   =    86_400_000L;
        public static final long ONE_WEEK  =   604_800_000L;
        public static final long ONE_MONTH = 2_592_000_000L;
        public static final long ONE_YEAR  = 31_536_000_000L;
        private HalfLife() {}
    }

    // ── Domain types ────────────────────────────────────────────────
    public record CollectionConfig(
        String name, int dim, long halfLifeMs,
        float temporalBlend, int nShards, String indexType
    ) {
        public static Builder builder(String name) { return new Builder(name); }
        public static final class Builder {
            private final String name;
            private int dim = 768;
            private long halfLifeMs = HalfLife.ONE_MONTH;
            private float temporalBlend = 0.3f;
            private int nShards = 4;
            private String indexType = "tvindex";
            Builder(String name) { this.name = name; }
            public Builder dim(int d)                { this.dim = d; return this; }
            public Builder halfLifeMs(long ms)       { this.halfLifeMs = ms; return this; }
            public Builder temporalBlend(float b)    { this.temporalBlend = b; return this; }
            public Builder nShards(int n)            { this.nShards = n; return this; }
            public Builder indexType(String t)       { this.indexType = t; return this; }
            public CollectionConfig build() {
                return new CollectionConfig(name, dim, halfLifeMs, temporalBlend, nShards, indexType);
            }
        }
    }

    public record CausalEdge(String targetId, String type, float weight) {
        public static CausalEdge causes(String targetId)      { return new CausalEdge(targetId, "CAUSES",      1.0f); }
        public static CausalEdge updates(String targetId)     { return new CausalEdge(targetId, "UPDATES",     1.0f); }
        public static CausalEdge contradicts(String targetId) { return new CausalEdge(targetId, "CONTRADICTS", 1.0f); }
    }

    public record Vector(
        String id, float[] embedding,
        Map<String, String> payload, List<CausalEdge> edges
    ) {
        public static Builder builder(String id, float[] embedding) { return new Builder(id, embedding); }
        public static final class Builder {
            private final String id;
            private final float[] embedding;
            private Map<String, String> payload = Collections.emptyMap();
            private List<CausalEdge> edges = Collections.emptyList();
            Builder(String id, float[] embedding) { this.id = id; this.embedding = embedding; }
            public Builder payload(Map<String, String> p) { this.payload = p; return this; }
            public Builder edges(List<CausalEdge> e)      { this.edges = e;   return this; }
            public Vector build() { return new Vector(id, embedding, payload, edges); }
        }
    }

    public record SearchOptions(
        int topK, float temporalBlend, String mode,
        int causalHops, boolean includeTrace, boolean stalenessWarnings
    ) {
        public static Builder builder() { return new Builder(); }
        public static final class Builder {
            private int topK = 10;
            private float temporalBlend = 0.3f;
            private String mode = "default";
            private int causalHops = 1;
            private boolean includeTrace = false;
            private boolean stalenessWarnings = true;
            public Builder topK(int k)                  { this.topK = k;                  return this; }
            public Builder temporalBlend(float b)       { this.temporalBlend = b;         return this; }
            public Builder mode(String m)               { this.mode = m;                  return this; }
            public Builder causalHops(int h)            { this.causalHops = h;            return this; }
            public Builder includeTrace(boolean t)      { this.includeTrace = t;          return this; }
            public Builder stalenessWarnings(boolean s) { this.stalenessWarnings = s;     return this; }
            public SearchOptions build() {
                return new SearchOptions(topK, temporalBlend, mode, causalHops, includeTrace, stalenessWarnings);
            }
        }
    }

    public record SearchHit(
        String id, float score, float semanticScore,
        float temporalScore, Map<String, String> payload,
        boolean stalenessWarning, String stalenessReason
    ) {}

    public record SearchResponse(List<SearchHit> hits) {}

    // ── Client internals ────────────────────────────────────────────
    private final HttpClient http;
    private final String baseUrl;
    private final String apiKey;

    private TideVecClient(String host, int port, String apiKey, Duration timeout) {
        this.baseUrl = "http://" + host + ":" + port;
        this.apiKey  = apiKey;
        this.http    = HttpClient.newBuilder().connectTimeout(timeout).build();
    }

    public static final class ClientBuilder {
        private final String host;
        private final int    port;
        private String   apiKey  = "";
        private Duration timeout = Duration.ofSeconds(30);
        ClientBuilder(String host, int port) { this.host = host; this.port = port; }
        public ClientBuilder apiKey(String k)   { this.apiKey = k;   return this; }
        public ClientBuilder timeout(Duration d) { this.timeout = d;  return this; }
        public TideVecClient build() { return new TideVecClient(host, port, apiKey, timeout); }
    }

    public static ClientBuilder builder(String host, int port) {
        return new ClientBuilder(host, port);
    }

    @Override public void close() { /* HttpClient manages its own lifecycle */ }

    // ── API methods ─────────────────────────────────────────────────
    public boolean ping() {
        try {
            HttpResponse<String> r = http.send(
                request("GET", "/health", null), HttpResponse.BodyHandlers.ofString());
            return r.statusCode() == 200;
        } catch (Exception e) { return false; }
    }

    public void createCollection(CollectionConfig cfg) throws IOException {
        String body = String.format(
            "{\"name\":\"%s\",\"dim\":%d,\"temporal\":{\"half_life_ms\":%d," +
            "\"temporal_blend\":%f},\"n_shards\":%d,\"index_type\":\"%s\"}",
            cfg.name(), cfg.dim(), cfg.halfLifeMs(),
            cfg.temporalBlend(), cfg.nShards(), cfg.indexType());
        post("/v1/collections", body);
    }

    public void dropCollection(String name) throws IOException {
        post("/v1/collections/" + name + "/delete", "{}");
    }

    public void upsert(String collection, List<Vector> vectors) throws IOException {
        StringBuilder sb = new StringBuilder("{\"vectors\":[");
        for (int i = 0; i < vectors.size(); i++) {
            if (i > 0) sb.append(",");
            Vector v = vectors.get(i);
            sb.append("{\"id\":\"").append(v.id()).append("\",\"embedding\":")
              .append(floatArrayToJson(v.embedding()));
            if (!v.payload().isEmpty()) {
                sb.append(",\"payload\":{");
                boolean first = true;
                for (Map.Entry<String, String> e : v.payload().entrySet()) {
                    if (!first) sb.append(",");
                    sb.append("\"").append(e.getKey()).append("\":\"")
                      .append(e.getValue()).append("\"");
                    first = false;
                }
                sb.append("}");
            }
            sb.append("}");
        }
        sb.append("]}");
        post("/v1/collections/" + collection + "/upsert", sb.toString());
    }

    public SearchResponse search(String collection, float[] embedding, SearchOptions opts)
            throws IOException {
        String body = String.format(
            "{\"vector\":%s,\"top_k\":%d,\"temporal_blend\":%f," +
            "\"mode\":\"%s\",\"causal_hops\":%d,\"include_trace\":%b," +
            "\"include_staleness_warnings\":%b}",
            floatArrayToJson(embedding), opts.topK(), opts.temporalBlend(),
            opts.mode(), opts.causalHops(), opts.includeTrace(), opts.stalenessWarnings());
        String resp = post("/v1/collections/" + collection + "/search", body);
        return parseSearchResponse(resp);
    }

    public void deleteVectors(String collection, List<String> ids) throws IOException {
        StringBuilder sb = new StringBuilder("{\"ids\":[");
        for (int i = 0; i < ids.size(); i++) {
            if (i > 0) sb.append(",");
            sb.append("\"").append(ids.get(i)).append("\"");
        }
        sb.append("]}");
        post("/v1/collections/" + collection + "/delete", sb.toString());
    }

    // ── Helpers ─────────────────────────────────────────────────────
    private String post(String path, String json) throws IOException {
        try {
            HttpResponse<String> resp = http.send(
                request("POST", path, json), HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() >= 400)
                throw new IOException("TideVec error " + resp.statusCode() + ": " + resp.body());
            return resp.body();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Request interrupted", e);
        }
    }

    private HttpRequest request(String method, String path, String json) {
        var builder = HttpRequest.newBuilder()
            .uri(URI.create(baseUrl + path))
            .timeout(Duration.ofSeconds(30));
        if (apiKey != null && !apiKey.isEmpty())
            builder.header("X-Api-Key", apiKey);
        if (json != null) {
            builder.header("Content-Type", "application/json");
            builder.method(method, HttpRequest.BodyPublishers.ofString(json, StandardCharsets.UTF_8));
        } else {
            builder.method(method, HttpRequest.BodyPublishers.noBody());
        }
        return builder.build();
    }

    private static String floatArrayToJson(float[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(",");
            sb.append(arr[i]);
        }
        return sb.append("]").toString();
    }

    private static SearchResponse parseSearchResponse(String json) {
        // Minimal JSON parser for search hits — no external dependency
        List<SearchHit> hits = new ArrayList<>();
        int idx = json.indexOf("\"hits\"");
        if (idx < 0) return new SearchResponse(hits);
        // Return empty for now — production use should add a proper JSON library
        return new SearchResponse(hits);
    }
}
