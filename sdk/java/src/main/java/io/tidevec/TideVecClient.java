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

    public record DriftStatus(
        String collection, String phase,
        int totalVectors, int migrated, int skipped,
        double pctComplete, String error
    ) {}

    public record CollectionInfo(
        String name, int nVectors, int nShards,
        int dim, String indexType, String metric, String backend
    ) {}

    // ── Client internals ────────────────────────────────────────────
    private final HttpClient http;
    private final String baseUrl;
    private final String apiKey;

    private TideVecClient(String host, int port, String apiKey, Duration timeout, boolean tls) {
        this.baseUrl = (tls ? "https" : "http") + "://" + host + ":" + port;
        this.apiKey  = (apiKey != null && !apiKey.isEmpty())
            ? apiKey
            : System.getenv("TIDEVEC_API_KEY") != null
                ? System.getenv("TIDEVEC_API_KEY") : "";
        this.http    = HttpClient.newBuilder().connectTimeout(timeout).build();
    }

    public static final class ClientBuilder {
        private final String host;
        private final int    port;
        private String   apiKey  = "";
        private Duration timeout = Duration.ofSeconds(30);
        private boolean  tls     = false;
        ClientBuilder(String host, int port) { this.host = host; this.port = port; }
        public ClientBuilder apiKey(String k)   { this.apiKey = k;   return this; }
        public ClientBuilder timeout(Duration d) { this.timeout = d;  return this; }
        public ClientBuilder tls(boolean t)    { this.tls = t;      return this; }
        public TideVecClient build() { return new TideVecClient(host, port, apiKey, timeout, tls); }
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

    /** Start zero-downtime embedding model migration. */
    public void startDrift(String collection, String reembedUrl) throws IOException {
        String body = String.format("{\"reembed_url\":\"%s\"}", reembedUrl);
        post("/v1/collections/" + collection + "/drift/start", body);
    }

    /** Poll migration progress. */
    public DriftStatus driftStatus(String collection) throws IOException {
        String resp = get("/v1/collections/" + collection + "/drift/status");
        return new DriftStatus(
            strField(resp, "collection"), strField(resp, "phase"),
            intField(resp, "total_vectors"), intField(resp, "migrated"),
            intField(resp, "skipped"), floatField(resp, "pct_complete"),
            strField(resp, "error") != null ? strField(resp, "error") : ""
        );
    }

    /** Abort an in-progress migration. */
    public void abortDrift(String collection) throws IOException {
        post("/v1/collections/" + collection + "/drift/abort", "{}");
    }

    /** Trigger a manual backup snapshot (requires auth). */
    public String triggerBackup() throws IOException {
        String resp = post("/v1/admin/backup", "{}");
        return strField(resp, "snapshot");
    }

    /** List available backup snapshots. */
    public List<String> listBackups() throws IOException {
        String resp = get("/v1/admin/backups");
        List<String> out = new ArrayList<>();
        int idx = resp.indexOf("\"snapshots\"");
        if (idx < 0) return out;
        int arrStart = resp.indexOf('[', idx);
        int arrEnd   = resp.indexOf(']', arrStart);
        if (arrStart < 0 || arrEnd < 0) return out;
        String arr = resp.substring(arrStart + 1, arrEnd);
        for (String part : arr.split(",")) {
            part = part.trim();
            if (part.startsWith("\"") && part.endsWith("\""))
                out.add(part.substring(1, part.length() - 1));
        }
        return out;
    }

    /** Restore from a snapshot (PITR). Server restart required. */
    public String restoreBackup(String snapshot) throws IOException {
        String body = String.format(
            "{\"snapshot\":\"%s\",\"confirm\":true}", snapshot);
        String resp = post("/v1/admin/restore", body);
        return strField(resp, "message");
    }

    // ── Helpers ─────────────────────────────────────────────────────
    private String get(String path) throws IOException {
        try {
            HttpResponse<String> resp = http.send(
                request("GET", path, null), HttpResponse.BodyHandlers.ofString());
            if (resp.statusCode() >= 400)
                throw new IOException("TideVec error " + resp.statusCode() + ": " + resp.body());
            return resp.body();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Request interrupted", e);
        }
    }

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
        List<SearchHit> hits = new ArrayList<>();

        // Navigate to data.results array
        // Server returns: {"status":"ok","data":{"results":[...],"count":N}}
        int dataIdx = json.indexOf("\"data\"");
        if (dataIdx < 0) return new SearchResponse(hits);

        int resultsIdx = json.indexOf("\"results\"", dataIdx);
        if (resultsIdx < 0) return new SearchResponse(hits);

        int arrStart = json.indexOf('[', resultsIdx);
        if (arrStart < 0) return new SearchResponse(hits);

        int arrEnd = findMatchingBracket(json, arrStart, '[', ']');
        if (arrEnd < 0) return new SearchResponse(hits);

        // Parse each object in the results array
        int pos = arrStart + 1;
        while (pos < arrEnd) {
            // Skip whitespace and commas
            while (pos < arrEnd && (json.charAt(pos) == ',' ||
                   Character.isWhitespace(json.charAt(pos)))) pos++;
            if (pos >= arrEnd) break;
            if (json.charAt(pos) != '{') { pos++; continue; }

            int objEnd = findMatchingBracket(json, pos, '{', '}');
            if (objEnd < 0) break;

            String obj = json.substring(pos, objEnd + 1);
            SearchHit hit = parseHit(obj);
            if (hit != null) hits.add(hit);
            pos = objEnd + 1;
        }
        return new SearchResponse(hits);
    }

    private static SearchHit parseHit(String obj) {
        String id              = strField(obj, "id");
        float  score           = floatField(obj, "score");
        float  semanticScore   = floatField(obj, "vector_score");
        float  temporalScore   = floatField(obj, "temporal_score");
        boolean stale          = boolField(obj, "staleness_warning");
        String  stalenessReason= strField(obj, "staleness_reason");

        // Parse payload object
        Map<String, String> payload = new LinkedHashMap<>();
        int pIdx = obj.indexOf("\"payload\"");
        if (pIdx >= 0) {
            int pStart = obj.indexOf('{', pIdx);
            if (pStart >= 0) {
                int pEnd = findMatchingBracket(obj, pStart, '{', '}');
                if (pEnd >= 0) {
                    String payloadStr = obj.substring(pStart + 1, pEnd);
                    // Parse key:value string pairs
                    int i = 0;
                    while (i < payloadStr.length()) {
                        while (i < payloadStr.length() &&
                               (payloadStr.charAt(i) == ',' ||
                                Character.isWhitespace(payloadStr.charAt(i)))) i++;
                        if (i >= payloadStr.length()) break;
                        if (payloadStr.charAt(i) != '"') { i++; continue; }
                        int kEnd = payloadStr.indexOf('"', i + 1);
                        if (kEnd < 0) break;
                        String key = payloadStr.substring(i + 1, kEnd);
                        i = kEnd + 1;
                        int colon = payloadStr.indexOf(':', i);
                        if (colon < 0) break;
                        i = colon + 1;
                        while (i < payloadStr.length() &&
                               Character.isWhitespace(payloadStr.charAt(i))) i++;
                        if (i >= payloadStr.length()) break;
                        if (payloadStr.charAt(i) == '"') {
                            // String value
                            int vEnd = i + 1;
                            while (vEnd < payloadStr.length()) {
                                if (payloadStr.charAt(vEnd) == '\\') { vEnd += 2; continue; }
                                if (payloadStr.charAt(vEnd) == '"') break;
                                vEnd++;
                            }
                            payload.put(key, payloadStr.substring(i + 1, vEnd));
                            i = vEnd + 1;
                        } else {
                            // Non-string value (number, bool, null)
                            int vEnd = i;
                            while (vEnd < payloadStr.length() &&
                                   payloadStr.charAt(vEnd) != ',' &&
                                   payloadStr.charAt(vEnd) != '}') vEnd++;
                            payload.put(key, payloadStr.substring(i, vEnd).trim());
                            i = vEnd;
                        }
                    }
                }
            }
        }

        if (id == null || id.isEmpty()) return null;
        return new SearchHit(id, score, semanticScore, temporalScore,
                             payload, stale, stalenessReason != null ? stalenessReason : "");
    }

    /** Find the closing bracket/brace matching the one at {@code start}. */
    private static int findMatchingBracket(String s, int start, char open, char close) {
        int depth = 0;
        boolean inStr = false;
        for (int i = start; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '\\' && inStr) { i++; continue; }
            if (c == '"') { inStr = !inStr; continue; }
            if (inStr) continue;
            if (c == open)  { depth++; }
            if (c == close) { depth--; if (depth == 0) return i; }
        }
        return -1;
    }

    private static String strField(String obj, String key) {
        String pat = "\"" + key + "\"";
        int idx = obj.indexOf(pat);
        if (idx < 0) return null;
        int colon = obj.indexOf(':', idx + pat.length());
        if (colon < 0) return null;
        int q1 = obj.indexOf('"', colon + 1);
        if (q1 < 0) return null;
        int q2 = q1 + 1;
        while (q2 < obj.length()) {
            if (obj.charAt(q2) == '\\') { q2 += 2; continue; }
            if (obj.charAt(q2) == '"') break;
            q2++;
        }
        return obj.substring(q1 + 1, q2);
    }

    private static float floatField(String obj, String key) {
        String pat = "\"" + key + "\"";
        int idx = obj.indexOf(pat);
        if (idx < 0) return 0f;
        int colon = obj.indexOf(':', idx + pat.length());
        if (colon < 0) return 0f;
        int start = colon + 1;
        while (start < obj.length() && Character.isWhitespace(obj.charAt(start))) start++;
        int end = start;
        while (end < obj.length() && (Character.isDigit(obj.charAt(end)) ||
               obj.charAt(end) == '.' || obj.charAt(end) == '-' ||
               obj.charAt(end) == 'E' || obj.charAt(end) == 'e')) end++;
        try { return Float.parseFloat(obj.substring(start, end)); }
        catch (NumberFormatException e) { return 0f; }
    }

    private static int intField(String obj, String key) {
        return (int) floatField(obj, key);
    }

    private static boolean boolField(String obj, String key) {
        String pat = "\"" + key + "\"";
        int idx = obj.indexOf(pat);
        if (idx < 0) return false;
        int colon = obj.indexOf(':', idx + pat.length());
        if (colon < 0) return false;
        int start = colon + 1;
        while (start < obj.length() && Character.isWhitespace(obj.charAt(start))) start++;
        return obj.startsWith("true", start);
    }
}
