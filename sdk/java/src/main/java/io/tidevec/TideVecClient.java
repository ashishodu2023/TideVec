package io.tidevec;

import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.*;

/**
 * TideVec Java Client — REST/HTTP implementation with zero runtime
 * dependencies (uses {@code java.net.http}).
 *
 * <h3>Quick Start:</h3>
 * <pre>{@code
 * try (TideVecClient db = TideVecClient.builder("localhost", 6399).build()) {
 *
 *     db.createCollection(CollectionConfig.builder("docs")
 *         .dim(768)
 *         .halfLifeMs(HalfLife.ONE_WEEK)
 *         .temporalBlend(0.3f)
 *         .supersession(0.92f, "doc_key")
 *         .build());
 *
 *     db.upsert("docs", List.of(
 *         Vector.builder("v1", new float[]{0.1f, 0.2f, 0.3f})
 *             .payload(Map.of("doc_key", "refund-policy"))
 *             .build()
 *     ));
 *
 *     SearchResponse results = db.search("docs", queryVec,
 *         SearchOptions.builder().topK(10).build());
 *
 *     for (SearchHit hit : results.hits()) {
 *         System.out.println(hit.id() + " score=" + hit.score()
 *             + (hit.isSuperseded() ? " (superseded by " + hit.supersededBy() + ")" : ""));
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

    /**
     * Collection settings. {@code supersedeThreshold} is compared against the
     * engine's vector score, which for cosine is (1+cos)/2: 0.92 ≈ cos 0.84.
     */
    public record CollectionConfig(
        String name, int dim, long halfLifeMs,
        float temporalBlend, int nShards, String indexType,
        boolean supersession, float supersedeThreshold, String supersedeMatchField,
        String textField, boolean payloadIndex, double compactionRatio
    ) {
        public CollectionConfig(String name, int dim, long halfLifeMs, float temporalBlend,
                                int nShards, String indexType, boolean supersession,
                                float supersedeThreshold, String supersedeMatchField) {
            this(name, dim, halfLifeMs, temporalBlend, nShards, indexType, supersession,
                 supersedeThreshold, supersedeMatchField, "", true);
        }
        public CollectionConfig(String name, int dim, long halfLifeMs, float temporalBlend,
                                int nShards, String indexType, boolean supersession,
                                float supersedeThreshold, String supersedeMatchField,
                                String textField, boolean payloadIndex) {
            this(name, dim, halfLifeMs, temporalBlend, nShards, indexType, supersession,
                 supersedeThreshold, supersedeMatchField, textField, payloadIndex, 0);
        }
        public static Builder builder(String name) { return new Builder(name); }
        public static final class Builder {
            private final String name;
            private int dim = 768;
            private long halfLifeMs = HalfLife.ONE_MONTH;
            private float temporalBlend = 0.3f;
            private int nShards = 4;
            private String indexType = "tvindex";
            private boolean supersession = false;
            private float supersedeThreshold = 0.92f;
            private String supersedeMatchField = "";
            private String textField = "";
            private boolean payloadIndex = true;
            private double compactionRatio = 0;
            Builder(String name) { this.name = name; }
            public Builder dim(int d)                { this.dim = d; return this; }
            public Builder halfLifeMs(long ms)       { this.halfLifeMs = ms; return this; }
            public Builder temporalBlend(float b)    { this.temporalBlend = b; return this; }
            public Builder nShards(int n)            { this.nShards = n; return this; }
            public Builder indexType(String t)       { this.indexType = t; return this; }
            /** Enable automatic newer-version detection on upsert. */
            public Builder supersession(float threshold, String matchField) {
                this.supersession = true;
                this.supersedeThreshold = threshold;
                this.supersedeMatchField = matchField == null ? "" : matchField;
                return this;
            }
            /** Payload field indexed for BM25 keyword / hybrid search. */
            public Builder textField(String f)       { this.textField = f == null ? "" : f; return this; }
            /** Index every payload key for filtered search (default on). */
            public Builder payloadIndex(boolean on)  { this.payloadIndex = on; return this; }
            /**
             * Rebuild indexes in the background once deleted/re-embedded slots
             * reach this fraction of the live vectors (server default 1.0).
             */
            public Builder compactionRatio(double r) { this.compactionRatio = r; return this; }
            public CollectionConfig build() {
                return new CollectionConfig(name, dim, halfLifeMs, temporalBlend, nShards, indexType,
                                            supersession, supersedeThreshold, supersedeMatchField,
                                            textField, payloadIndex, compactionRatio);
            }
        }
    }

    public record CausalEdge(String targetId, String type, float weight) {
        public static CausalEdge causes(String targetId)      { return new CausalEdge(targetId, "CAUSES",      1.0f); }
        public static CausalEdge updates(String targetId)     { return new CausalEdge(targetId, "UPDATES",     1.0f); }
        public static CausalEdge contradicts(String targetId) { return new CausalEdge(targetId, "CONTRADICTS", 1.0f); }
    }

    /** A record to upsert. {@code createdAt} (ms since epoch) defaults to now when null. */
    public record Vector(
        String id, float[] embedding,
        Map<String, String> payload, List<CausalEdge> edges, Long createdAt
    ) {
        public static Builder builder(String id, float[] embedding) { return new Builder(id, embedding); }
        public static final class Builder {
            private final String id;
            private final float[] embedding;
            private Map<String, String> payload = Collections.emptyMap();
            private List<CausalEdge> edges = Collections.emptyList();
            private Long createdAt = null;
            Builder(String id, float[] embedding) { this.id = id; this.embedding = embedding; }
            public Builder payload(Map<String, String> p) { this.payload = p; return this; }
            public Builder edges(List<CausalEdge> e)      { this.edges = e;   return this; }
            public Builder createdAt(long ms)             { this.createdAt = ms; return this; }
            public Vector build() { return new Vector(id, embedding, payload, edges, createdAt); }
        }
    }

    /**
     * Search settings. A null {@code temporalBlend} uses the collection's
     * configured blend; 0 is semantic-only.
     */
    public record SearchOptions(
        int topK, Float temporalBlend, String mode,
        int causalHops, boolean includeTrace, boolean stalenessWarnings,
        float supersededPenalty, String filter, int efSearch,
        Map<String, Object> where, Long asOf, String text, String fusion, Float alpha
    ) {
        public SearchOptions(int topK, Float temporalBlend, String mode, int causalHops,
                             boolean includeTrace, boolean stalenessWarnings,
                             float supersededPenalty, String filter, int efSearch) {
            this(topK, temporalBlend, mode, causalHops, includeTrace, stalenessWarnings,
                 supersededPenalty, filter, efSearch, null, null, "", null, null);
        }
        public static Builder builder() { return new Builder(); }
        public static final class Builder {
            private int topK = 10;
            private Float temporalBlend = null;
            private String mode = "vector_only";
            private int causalHops = 1;
            private boolean includeTrace = false;
            private boolean stalenessWarnings = true;
            private float supersededPenalty = 0.0f;
            private String filter = "";
            private int efSearch = 128;
            private Map<String, Object> where = null;
            private Long asOf = null;
            private String text = "";
            private String fusion = null;
            private Float alpha = null;
            public Builder topK(int k)                  { this.topK = k;                  return this; }
            public Builder temporalBlend(float b)       { this.temporalBlend = b;         return this; }
            public Builder mode(String m)               { this.mode = m;                  return this; }
            public Builder causalHops(int h)            { this.causalHops = h;            return this; }
            public Builder includeTrace(boolean t)      { this.includeTrace = t;          return this; }
            public Builder stalenessWarnings(boolean s) { this.stalenessWarnings = s;     return this; }
            /** Scale down results that have a newer version (0 = annotate only, 1 = sink). */
            public Builder supersededPenalty(float p)   { this.supersededPenalty = p;     return this; }
            /** Exact-match payload filter, {@code "key=value"}. */
            public Builder filter(String f)             { this.filter = f == null ? "" : f; return this; }
            public Builder efSearch(int ef)             { this.efSearch = ef;             return this; }
            /** Rich filter (see {@link Filters}); takes precedence over {@link #filter}. */
            public Builder where(Map<String, Object> w) { this.where = w;                 return this; }
            /** Search the collection as of a time (ms since epoch). */
            public Builder asOf(long ms)                { this.asOf = ms;                 return this; }
            public Builder asOf(java.time.Instant t)    { this.asOf = t.toEpochMilli();   return this; }
            /**
             * BM25 keyword query over the collection's text field. With a null
             * embedding the search is keyword-only; otherwise it is hybrid.
             */
            public Builder text(String t)               { this.text = t == null ? "" : t; return this; }
            /** Hybrid fusion: {@code "rrf"} or {@code "weighted"}; alpha weights the vector side. */
            public Builder hybrid(String fusion, float alpha) {
                this.fusion = fusion;
                this.alpha = alpha;
                return this;
            }
            public SearchOptions build() {
                return new SearchOptions(topK, temporalBlend, mode, causalHops, includeTrace,
                                         stalenessWarnings, supersededPenalty, filter, efSearch,
                                         where, asOf, text, fusion, alpha);
            }
        }
    }

    public record SearchHit(
        String id, float score, float semanticScore,
        float temporalScore, long createdAt, Map<String, String> payload,
        boolean stalenessWarning, String stalenessReason,
        List<String> supersededBy, List<String> contradictedBy, List<String> causalNeighbors,
        float textScore
    ) {
        public SearchHit(String id, float score, float semanticScore, float temporalScore,
                         long createdAt, Map<String, String> payload, boolean stalenessWarning,
                         String stalenessReason, List<String> supersededBy,
                         List<String> contradictedBy, List<String> causalNeighbors) {
            this(id, score, semanticScore, temporalScore, createdAt, payload, stalenessWarning,
                 stalenessReason, supersededBy, contradictedBy, causalNeighbors, 0f);
        }
        /** True when a newer version of this record exists. */
        public boolean isSuperseded() { return !supersededBy.isEmpty(); }
    }

    /** A stored vector returned by get / scroll ({@code embedding} is null unless requested). */
    public record StoredVector(
        String id, float[] embedding, Map<String, String> payload,
        long createdAt, long validFrom,
        List<String> supersededBy, List<String> supersedes
    ) {
        public boolean isSuperseded() { return !supersededBy.isEmpty(); }
    }

    /** One page of a scroll; {@code nextCursor} is null on the last page. */
    public record ScrollPage(List<StoredVector> records, String nextCursor) {}

    /** Selects a page of records (in id order). */
    public record ScrollOptions(
        int limit, String cursor, String filter, Map<String, Object> where, Long asOf,
        boolean withVector, boolean withPayload
    ) {
        public static Builder builder() { return new Builder(); }
        public static final class Builder {
            private int limit = 100;
            private String cursor = null;
            private String filter = "";
            private Map<String, Object> where = null;
            private Long asOf = null;
            private boolean withVector = false;
            private boolean withPayload = true;
            public Builder limit(int n)                 { this.limit = n;       return this; }
            public Builder cursor(String c)             { this.cursor = c;      return this; }
            public Builder filter(String f)             { this.filter = f == null ? "" : f; return this; }
            public Builder where(Map<String, Object> w) { this.where = w;       return this; }
            public Builder asOf(long ms)                { this.asOf = ms;       return this; }
            public Builder withVector(boolean v)        { this.withVector = v;  return this; }
            public Builder withPayload(boolean p)       { this.withPayload = p; return this; }
            public ScrollOptions build() {
                return new ScrollOptions(limit, cursor, filter, where, asOf, withVector, withPayload);
            }
        }
    }

    /**
     * Builders for rich filter objects, e.g.
     * {@code Filters.and(Filters.eq("lang", "en"), Filters.range("year", 2020, null))}.
     * Plain {@code Map.of("lang", "en")} works too. Supported operators:
     * $eq $ne $in $nin $gt $gte $lt $lte $exists $and $or $not; ranges
     * accept numbers or ISO-8601 dates; {@code created_at} and {@code _id}
     * are always available.
     */
    public static final class Filters {
        private Filters() {}
        public static Map<String, Object> eq(String key, Object value) { return Map.of(key, value); }
        public static Map<String, Object> ne(String key, Object value) {
            return Map.of(key, Map.of("$ne", value));
        }
        public static Map<String, Object> in(String key, Collection<?> values) {
            return Map.of(key, Map.of("$in", List.copyOf(values)));
        }
        public static Map<String, Object> nin(String key, Collection<?> values) {
            return Map.of(key, Map.of("$nin", List.copyOf(values)));
        }
        /** {@code gte <= key < lt}; either bound may be null. */
        public static Map<String, Object> range(String key, Object gte, Object lt) {
            Map<String, Object> ops = new LinkedHashMap<>();
            if (gte != null) ops.put("$gte", gte);
            if (lt != null) ops.put("$lt", lt);
            return Map.of(key, ops);
        }
        public static Map<String, Object> exists(String key, boolean exists) {
            return Map.of(key, Map.of("$exists", exists));
        }
        @SafeVarargs
        public static Map<String, Object> and(Map<String, Object>... fs) { return Map.of("$and", List.of(fs)); }
        @SafeVarargs
        public static Map<String, Object> or(Map<String, Object>... fs)  { return Map.of("$or", List.of(fs)); }
        public static Map<String, Object> not(Map<String, Object> f)     { return Map.of("$not", f); }
    }

    public record SearchResponse(List<SearchHit> hits, int count) {}

    public record DriftStatus(
        String collection, String phase,
        int totalVectors, int migrated, int skipped,
        double pctComplete, String error
    ) {}

    /** Index compaction state of a collection; counts are summed over shards. */
    public record CompactionStatus(
        int started, boolean running, int runningShards, int runs,
        int tombstones, long indexBytes, int lastReclaimed, int lastReplayed,
        double lastMs
    ) {
        static CompactionStatus fromJson(Map<String, Object> d) {
            return new CompactionStatus(
                num(d.get("started")).intValue(), Boolean.TRUE.equals(d.get("running")),
                num(d.get("running_shards")).intValue(), num(d.get("runs")).intValue(),
                num(d.get("tombstones")).intValue(), num(d.get("index_bytes")).longValue(),
                num(d.get("last_reclaimed")).intValue(), num(d.get("last_replayed")).intValue(),
                num(d.get("last_ms")).doubleValue());
        }
    }

    public record CollectionInfo(
        String name, int nVectors, int nShards,
        int dim, String indexType, String metric, String backend,
        boolean supersessionEnabled, String textField, List<String> aliases
    ) {
        public CollectionInfo(String name, int nVectors, int nShards, int dim, String indexType,
                              String metric, String backend, boolean supersessionEnabled) {
            this(name, nVectors, nShards, dim, indexType, metric, backend, supersessionEnabled,
                 "", List.of());
        }
    }

    // ── Client internals ────────────────────────────────────────────
    private final HttpClient http;
    private final String baseUrl;
    private final String apiKey;
    private final Duration timeout;
    private final int maxRetries;

    private TideVecClient(String host, int port, String apiKey, Duration timeout, boolean tls,
                          int maxRetries) {
        this.baseUrl = (tls ? "https" : "http") + "://" + host + ":" + port;
        String env = System.getenv("TIDEVEC_API_KEY");
        this.apiKey  = (apiKey != null && !apiKey.isEmpty()) ? apiKey : (env != null ? env : "");
        this.timeout = timeout;
        this.maxRetries = maxRetries;
        this.http    = HttpClient.newBuilder().connectTimeout(timeout).build();
    }

    public static final class ClientBuilder {
        private final String host;
        private final int    port;
        private String   apiKey  = "";
        private Duration timeout = Duration.ofSeconds(30);
        private boolean  tls     = false;
        private int      maxRetries = 3;
        ClientBuilder(String host, int port) { this.host = host; this.port = port; }
        public ClientBuilder apiKey(String k)    { this.apiKey = k;   return this; }
        public ClientBuilder timeout(Duration d) { this.timeout = d;  return this; }
        public ClientBuilder tls(boolean t)      { this.tls = t;      return this; }
        /** Retries for 429 rate-limit responses. */
        public ClientBuilder maxRetries(int n)   { this.maxRetries = n; return this; }
        public TideVecClient build() { return new TideVecClient(host, port, apiKey, timeout, tls, maxRetries); }
    }

    public static ClientBuilder builder(String host, int port) {
        return new ClientBuilder(host, port);
    }

    @Override public void close() { /* HttpClient manages its own lifecycle */ }

    // ── API methods ─────────────────────────────────────────────────
    public boolean ping() {
        try {
            call("GET", "/health", null);
            return true;
        } catch (IOException e) {
            return false;
        }
    }

    /** Server metadata from GET /v1/info. */
    public Map<String, Object> info() throws IOException {
        return asMap(call("GET", "/v1/info", null));
    }

    public void createCollection(CollectionConfig cfg) throws IOException {
        Map<String, Object> temporal = new LinkedHashMap<>();
        temporal.put("half_life_ms", cfg.halfLifeMs());
        temporal.put("temporal_blend", cfg.temporalBlend());
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("name", cfg.name());
        body.put("dim", cfg.dim());
        body.put("n_shards", cfg.nShards());
        body.put("index_type", cfg.indexType());
        body.put("temporal", temporal);
        if (cfg.supersession()) {
            Map<String, Object> sup = new LinkedHashMap<>();
            sup.put("similarity_threshold", cfg.supersedeThreshold());
            sup.put("match_field", cfg.supersedeMatchField());
            body.put("supersession", sup);
        }
        if (cfg.textField() != null && !cfg.textField().isEmpty()) body.put("text_field", cfg.textField());
        if (!cfg.payloadIndex()) body.put("payload_index", false);
        if (cfg.compactionRatio() > 0) body.put("compaction_ratio", cfg.compactionRatio());
        call("POST", "/v1/collections", body);
    }

    public void dropCollection(String name) throws IOException {
        call("DELETE", "/v1/collections/" + enc(name), null);
    }

    public CollectionInfo getCollection(String name) throws IOException {
        return collectionInfo(data(call("GET", "/v1/collections/" + enc(name), null)));
    }

    public List<CollectionInfo> listCollections() throws IOException {
        List<CollectionInfo> out = new ArrayList<>();
        for (Object o : asList(asMap(call("GET", "/v1/collections", null)).get("data")))
            out.add(collectionInfo(asMap(o)));
        return out;
    }

    /** Number of live vectors in a collection. */
    public int count(String collection) throws IOException {
        return getCollection(collection).nVectors();
    }

    public void upsert(String collection, List<Vector> vectors) throws IOException {
        List<Object> items = new ArrayList<>();
        for (Vector v : vectors) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", v.id());
            m.put("embedding", v.embedding());
            if (v.payload() != null && !v.payload().isEmpty()) m.put("payload", v.payload());
            if (v.edges() != null && !v.edges().isEmpty()) {
                List<Object> edges = new ArrayList<>();
                for (CausalEdge e : v.edges())
                    edges.add(Map.of("target_id", e.targetId(), "type", e.type(), "weight", e.weight()));
                m.put("edges", edges);
            }
            if (v.createdAt() != null) m.put("created_at", v.createdAt());
            items.add(m);
        }
        call("POST", col(collection, "upsert"), Map.of("vectors", items));
    }

    public SearchResponse search(String collection, float[] embedding, SearchOptions opts)
            throws IOException {
        boolean hasText = opts.text() != null && !opts.text().isEmpty();
        if (embedding == null && !hasText)
            throw new IllegalArgumentException("search needs an embedding or text");
        Map<String, Object> body = new LinkedHashMap<>();
        if (embedding != null) body.put("vector", embedding);
        body.put("top_k", opts.topK());
        if (opts.temporalBlend() != null) body.put("temporal_blend", opts.temporalBlend());
        body.put("mode", opts.mode());
        body.put("causal_hops", opts.causalHops());
        body.put("include_trace", opts.includeTrace());
        body.put("include_staleness_warnings", opts.stalenessWarnings());
        body.put("superseded_penalty", opts.supersededPenalty());
        body.put("ef_search", opts.efSearch());
        putFilter(body, opts.filter(), opts.where(), opts.asOf());
        if (hasText) {
            body.put("text", opts.text());
            if (opts.fusion() != null || opts.alpha() != null) {
                Map<String, Object> hybrid = new LinkedHashMap<>();
                hybrid.put("fusion", opts.fusion() == null ? "rrf" : opts.fusion());
                if (opts.alpha() != null) hybrid.put("alpha", opts.alpha());
                body.put("hybrid", hybrid);
            }
        }

        Map<String, Object> data = data(call("POST", col(collection, "search"), body));
        List<SearchHit> hits = new ArrayList<>();
        for (Object o : asList(data.get("results"))) hits.add(searchHit(asMap(o)));
        return new SearchResponse(hits, num(data.get("count")).intValue());
    }

    /**
     * Delete by id list and/or a {@code "key=value"} payload filter (pass
     * null to skip either). Returns how many vectors were removed.
     */
    public int delete(String collection, List<String> ids, String filter) throws IOException {
        if (ids == null && (filter == null || filter.isEmpty()))
            throw new IllegalArgumentException("delete needs ids or a filter");
        Map<String, Object> body = new LinkedHashMap<>();
        if (ids != null) body.put("ids", ids);
        if (filter != null && !filter.isEmpty()) {
            int eq = filter.indexOf('=');
            if (eq <= 0) throw new IllegalArgumentException("filter must look like key=value: " + filter);
            body.put("filter", filter);
        }
        return num(data(call("POST", col(collection, "delete"), body)).get("deleted")).intValue();
    }

    /** Delete every vector whose payload matches {@code "key=value"}. */
    public int deleteWhere(String collection, String filter) throws IOException {
        return delete(collection, null, filter);
    }

    /** Delete every vector matching a rich filter (see {@link Filters}). */
    public int deleteWhere(String collection, Map<String, Object> where) throws IOException {
        if (where == null || where.isEmpty())
            throw new IllegalArgumentException("deleteWhere needs a non-empty filter");
        return num(data(call("POST", col(collection, "delete"), Map.of("filter", where)))
                   .get("deleted")).intValue();
    }

    /** Delete vectors by id; returns how many existed and were removed. */
    public int deleteVectors(String collection, List<String> ids) throws IOException {
        return delete(collection, ids, null);
    }

    /** Fetch one stored vector by id. */
    public Optional<StoredVector> get(String collection, String id, boolean withVector) throws IOException {
        List<StoredVector> found = getMany(collection, List.of(id), withVector);
        return found.isEmpty() ? Optional.empty() : Optional.of(found.get(0));
    }

    /** Fetch stored vectors by id, in request order; missing ids are skipped. */
    public List<StoredVector> getMany(String collection, List<String> ids, boolean withVector)
            throws IOException {
        Map<String, Object> d = data(call("POST", col(collection, "get"),
                                          Map.of("ids", ids, "with_vector", withVector)));
        List<StoredVector> out = new ArrayList<>();
        for (Object o : asList(d.get("vectors"))) out.add(storedVector(asMap(o)));
        return out;
    }

    /** One page of records in id order; pass {@code nextCursor} back as the cursor. */
    public ScrollPage scroll(String collection, ScrollOptions opts) throws IOException {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("limit", opts.limit());
        if (opts.cursor() != null) body.put("cursor", opts.cursor());
        body.put("with_vector", opts.withVector());
        body.put("with_payload", opts.withPayload());
        putFilter(body, opts.filter(), opts.where(), opts.asOf());
        Map<String, Object> d = data(call("POST", col(collection, "scroll"), body));
        List<StoredVector> recs = new ArrayList<>();
        for (Object o : asList(d.get("vectors"))) recs.add(storedVector(asMap(o)));
        return new ScrollPage(recs, d.get("next_cursor") instanceof String s ? s : null);
    }

    /** Every (matching) id in order, starting after {@code opts.cursor()}. */
    public List<String> listIds(String collection, ScrollOptions opts) throws IOException {
        List<String> ids = new ArrayList<>();
        String cursor = opts.cursor();
        while (true) {
            ScrollPage page = scroll(collection, new ScrollOptions(
                opts.limit(), cursor, opts.filter(), opts.where(), opts.asOf(), false, false));
            for (StoredVector r : page.records()) ids.add(r.id());
            if (page.nextCursor() == null) return ids;
            cursor = page.nextCursor();
        }
    }

    /** Count vectors matching {@code where} (null = all) created by {@code asOf} (null = now). */
    public int count(String collection, Map<String, Object> where, Long asOf) throws IOException {
        Map<String, Object> body = new LinkedHashMap<>();
        putFilter(body, "", where, asOf);
        return num(data(call("POST", col(collection, "count"), body)).get("count")).intValue();
    }

    /**
     * Point {@code alias} at {@code collection}, atomically repointing it if it
     * exists. Every collection method accepts the alias as a name. Returns the
     * collection it pointed at before, or null.
     */
    public String setAlias(String alias, String collection) throws IOException {
        Object prev = data(call("PUT", "/v1/aliases/" + enc(alias),
                                Map.of("collection", collection))).get("previous");
        return prev instanceof String s ? s : null;
    }

    public void deleteAlias(String alias) throws IOException {
        call("DELETE", "/v1/aliases/" + enc(alias), null);
    }

    /** alias → collection. */
    public Map<String, String> listAliases() throws IOException {
        Map<String, String> out = new LinkedHashMap<>();
        for (Object o : asList(data(call("GET", "/v1/aliases", null)).get("aliases"))) {
            Map<String, Object> a = asMap(o);
            out.put(str(a.get("alias")), str(a.get("collection")));
        }
        return out;
    }

    /**
     * Rebuild the collection's vector, payload and text indexes without the
     * tombstones left by deletes and re-embeds. The rebuild runs in the
     * background while reads and writes continue; with {@code wait} the call
     * returns once the rebuilt indexes are in place.
     */
    public CompactionStatus compact(String collection, boolean wait) throws IOException {
        return CompactionStatus.fromJson(data(call("POST", col(collection, "compact"),
                                                   Map.of("wait", wait))));
    }

    public CompactionStatus compactionStatus(String collection) throws IOException {
        return CompactionStatus.fromJson(data(call("GET", col(collection, "compact"), null)));
    }

    /** Add typed edges between existing vectors. */
    public int addEdges(String collection, List<Edge> edges) throws IOException {
        List<Object> list = new ArrayList<>();
        for (Edge e : edges)
            list.add(Map.of("src", e.getSrc(), "tgt", e.getTgt(), "type", e.getType(),
                            "weight", e.getWeight()));
        return num(data(call("POST", col(collection, "edges"), Map.of("edges", list)))
                   .get("added")).intValue();
    }

    /** Update a collection's decay half-life and default blend. */
    public void setTemporal(String collection, long halfLifeMs, float temporalBlend) throws IOException {
        call("PUT", col(collection, "temporal"),
             Map.of("half_life_ms", halfLifeMs, "temporal_blend", temporalBlend));
    }

    /**
     * Raft status of this node (experimental cluster mode). Contains
     * {@code mode = "standalone"} on a single server.
     */
    public Map<String, Object> clusterStatus() throws IOException {
        return data(call("GET", "/v1/cluster/status", null));
    }

    /** Start zero-downtime embedding model migration. */
    public void startDrift(String collection, String reembedUrl) throws IOException {
        call("POST", col(collection, "drift/start"), Map.of("reembed_url", reembedUrl));
    }

    /** Poll migration progress. */
    public DriftStatus driftStatus(String collection) throws IOException {
        Map<String, Object> d = data(call("GET", col(collection, "drift/status"), null));
        return new DriftStatus(
            str(d.get("collection")), str(d.get("phase")),
            num(d.get("total_vectors")).intValue(), num(d.get("migrated")).intValue(),
            num(d.get("skipped")).intValue(), num(d.get("pct_complete")).doubleValue(),
            str(d.get("error")));
    }

    /** Abort an in-progress migration. */
    public void abortDrift(String collection) throws IOException {
        call("POST", col(collection, "drift/abort"), Map.of());
    }

    /** Trigger a manual backup snapshot (requires auth). */
    public String triggerBackup() throws IOException {
        return str(data(call("POST", "/v1/admin/backup", Map.of())).get("snapshot"));
    }

    /** List available backup snapshots. */
    public List<String> listBackups() throws IOException {
        return strList(data(call("GET", "/v1/admin/backups", null)).get("snapshots"));
    }

    /** Restore from a snapshot (PITR). Server restart required. */
    public String restoreBackup(String snapshot) throws IOException {
        return str(data(call("POST", "/v1/admin/restore",
                             Map.of("snapshot", snapshot, "confirm", true))).get("message"));
    }

    // ── HTTP ────────────────────────────────────────────────────────

    /**
     * Sends a request, retrying 429s and following cluster-mode leader
     * redirects (421). Returns the parsed response body.
     */
    private Object call(String method, String path, Object body) throws IOException {
        String json = body == null ? null : Json.write(body);
        String base = baseUrl;
        int rateRetries = 0, hops = 0;
        long redirectDeadline = System.nanoTime() + Duration.ofSeconds(10).toNanos();
        while (true) {
            HttpResponse<String> resp;
            try {
                resp = http.send(request(method, base + path, json), HttpResponse.BodyHandlers.ofString());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("Request interrupted", e);
            } catch (IOException e) {
                if (!base.equals(baseUrl) && System.nanoTime() < redirectDeadline) {
                    base = baseUrl;   // stale leader hint pointing at a dead node
                    continue;
                }
                throw e;
            }
            int status = resp.statusCode();
            if (status < 400) return resp.body().isEmpty() ? Map.of() : Json.parse(resp.body());

            String message = resp.body();
            String leader = null;
            try {
                Map<String, Object> err = asMap(Json.parse(resp.body()));
                if (err.get("error") instanceof String m) message = m;
                if (err.get("leader") instanceof String l) leader = l;
            } catch (IllegalArgumentException ignored) { /* non-JSON error body */ }

            if (status == 429 && rateRetries < maxRetries) {
                rateRetries++;
                sleep(Math.min(rateRetries * rateRetries, 8) * 1000L);
            } else if (status == 421 && System.nanoTime() < redirectDeadline) {
                hops++;
                String hint = leader == null ? null : leader.replaceAll("/+$", "");
                if (hint != null && !hint.equals(base) && hops <= 2) {
                    base = hint;
                    continue;
                }
                sleep(Math.min(50L * hops, 500L));
                base = hint != null ? hint : baseUrl;
            } else {
                throw new TideVecException(status, message, leader);
            }
        }
    }

    private HttpRequest request(String method, String url, String json) {
        var builder = HttpRequest.newBuilder().uri(URI.create(url)).timeout(timeout);
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

    private static void sleep(long ms) throws IOException {
        try {
            Thread.sleep(ms);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Request interrupted", e);
        }
    }

    // ── Decoding helpers ────────────────────────────────────────────

    private static String enc(String s) {
        return URLEncoder.encode(s, StandardCharsets.UTF_8).replace("+", "%20");
    }

    private static String col(String collection, String op) {
        return "/v1/collections/" + enc(collection) + "/" + op;
    }

    private static Map<String, Object> data(Object resp) {
        return asMap(asMap(resp).get("data"));
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(Object o) {
        return o instanceof Map<?, ?> m ? (Map<String, Object>) m : Map.of();
    }

    @SuppressWarnings("unchecked")
    private static List<Object> asList(Object o) {
        return o instanceof List<?> l ? (List<Object>) l : List.of();
    }

    private static Number num(Object o) {
        return o instanceof Number n ? n : 0;
    }

    private static String str(Object o) {
        return o instanceof String s ? s : "";
    }

    private static List<String> strList(Object o) {
        List<String> out = new ArrayList<>();
        for (Object x : asList(o)) if (x instanceof String s) out.add(s);
        return out;
    }

    private static void putFilter(Map<String, Object> body, String filter,
                                  Map<String, Object> where, Long asOf) {
        if (where != null) {
            body.put("filter", where);
        } else if (filter != null && !filter.isEmpty()) {
            if (filter.indexOf('=') <= 0)
                throw new IllegalArgumentException("filter must look like key=value: " + filter);
            body.put("filter", filter);
        }
        if (asOf != null) body.put("as_of", asOf);
    }

    private static CollectionInfo collectionInfo(Map<String, Object> m) {
        return new CollectionInfo(
            str(m.get("name")), num(m.get("n_vectors")).intValue(), num(m.get("n_shards")).intValue(),
            num(m.get("dim")).intValue(), str(m.get("index_type")), str(m.get("metric")),
            str(m.get("backend")),
            Boolean.TRUE.equals(asMap(m.get("supersession")).get("enabled")),
            str(m.get("text_field")), strList(m.get("aliases")));
    }

    private static Map<String, String> payload(Object o) {
        Map<String, String> payload = new LinkedHashMap<>();
        for (Map.Entry<String, Object> e : asMap(o).entrySet())
            payload.put(e.getKey(), e.getValue() instanceof String s ? s : String.valueOf(e.getValue()));
        return payload;
    }

    private static SearchHit searchHit(Map<String, Object> m) {
        return new SearchHit(
            str(m.get("id")), num(m.get("score")).floatValue(),
            num(m.get("vector_score")).floatValue(), num(m.get("temporal_score")).floatValue(),
            num(m.get("created_at")).longValue(), payload(m.get("payload")),
            Boolean.TRUE.equals(m.get("staleness_warning")), str(m.get("staleness_reason")),
            strList(m.get("superseded_by")), strList(m.get("contradicted_by")),
            strList(m.get("causal_neighbors")), num(m.get("text_score")).floatValue());
    }

    private static StoredVector storedVector(Map<String, Object> m) {
        float[] emb = null;
        if (m.get("embedding") instanceof List<?> l) {
            emb = new float[l.size()];
            for (int i = 0; i < emb.length; i++) emb[i] = num(l.get(i)).floatValue();
        }
        return new StoredVector(
            str(m.get("id")), emb, payload(m.get("payload")),
            num(m.get("created_at")).longValue(), num(m.get("valid_from")).longValue(),
            strList(m.get("superseded_by")), strList(m.get("supersedes")));
    }
}
