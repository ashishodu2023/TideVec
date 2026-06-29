package io.tidevec;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Metadata;
import io.grpc.stub.MetadataUtils;
import io.grpc.stub.StreamObserver;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * TideVec Java Client.
 *
 * <p>Temporally-aware causal vector database SDK.
 *
 * <h3>Quick Start:</h3>
 * <pre>{@code
 * try (TideVecClient db = TideVecClient.builder("localhost", 6399).build()) {
 *
 *     // Create collection
 *     db.createCollection(CollectionConfig.builder("docs")
 *         .dim(768)
 *         .halfLifeMs(HalfLife.ONE_WEEK)
 *         .temporalBlend(0.3f)
 *         .build());
 *
 *     // Upsert vectors
 *     db.upsert("docs", List.of(
 *         Vector.builder("v1", embedding)
 *             .payload("source", "wiki")
 *             .ttlSeconds(86400)
 *             .build()
 *     ));
 *
 *     // Search with temporal scoring
 *     SearchResponse resp = db.search("docs", SearchRequest.builder(queryEmbedding)
 *         .topK(10)
 *         .temporalBlend(0.3f)
 *         .includeStalenessWarnings(true)
 *         .build());
 *
 *     for (SearchHit hit : resp.getHits()) {
 *         System.out.printf("%s  score=%.4f  temporal=%.3f%n",
 *             hit.getId(), hit.getScore(), hit.getTemporalScore());
 *     }
 * }
 * }</pre>
 */
public class TideVecClient implements AutoCloseable {

    // ================================================================
    // Half-life constants (milliseconds)
    // ================================================================
    public static final class HalfLife {
        public static final long ONE_HOUR   = 3_600_000L;
        public static final long ONE_DAY    = 86_400_000L;
        public static final long ONE_WEEK   = 604_800_000L;
        public static final long ONE_MONTH  = 2_592_000_000L;
        public static final long ONE_YEAR   = 31_536_000_000L;

        public static final long AGENT_SESSION  = ONE_HOUR;
        public static final long NEWS_FEED      = ONE_DAY;
        public static final long SUPPORT_TICKET = ONE_MONTH;
        public static final long DOCUMENT_STORE = ONE_YEAR;

        private HalfLife() {}
    }

    // ================================================================
    // Domain types
    // ================================================================

    public enum EdgeType {
        CAUSES, CONTRADICTS, UPDATES, RELATED_TO, ENTITY_OF, SUPPORTS
    }

    public enum QueryMode {
        VECTOR_ONLY, CAUSAL_EXPAND, CONTRADICTION_CHECK, ENTITY_RESOLVE
    }

    public enum Device { AUTO, CPU, GPU, TPU }

    // ---- CausalEdge --------------------------------------------
    public static final class CausalEdge {
        private final String   targetId;
        private final EdgeType type;
        private final float    weight;

        public CausalEdge(String targetId, EdgeType type, float weight) {
            this.targetId = targetId;
            this.type     = type;
            this.weight   = weight;
        }

        public String   getTargetId() { return targetId; }
        public EdgeType getType()     { return type; }
        public float    getWeight()   { return weight; }
    }

    // ---- Vector ------------------------------------------------
    public static final class Vector {
        private final String              id;
        private final float[]             embedding;
        private final Map<String, String> payload;
        private final long                ttlSeconds;
        private final List<CausalEdge>    edges;

        private Vector(Builder b) {
            this.id         = b.id;
            this.embedding  = b.embedding;
            this.payload    = Collections.unmodifiableMap(b.payload);
            this.ttlSeconds = b.ttlSeconds;
            this.edges      = Collections.unmodifiableList(b.edges);
        }

        public static Builder builder(String id, float[] embedding) {
            return new Builder(id, embedding);
        }

        public String              getId()        { return id; }
        public float[]             getEmbedding() { return embedding; }
        public Map<String, String> getPayload()   { return payload; }
        public long                getTtlSeconds(){ return ttlSeconds; }
        public List<CausalEdge>    getEdges()     { return edges; }

        public static final class Builder {
            private final String              id;
            private final float[]             embedding;
            private final Map<String, String> payload = new LinkedHashMap<>();
            private long                      ttlSeconds = 0;
            private final List<CausalEdge>    edges = new ArrayList<>();

            Builder(String id, float[] embedding) {
                this.id = id;
                this.embedding = embedding;
            }
            public Builder payload(String key, String value) {
                payload.put(key, value); return this;
            }
            public Builder payload(Map<String, String> map) {
                payload.putAll(map); return this;
            }
            public Builder ttlSeconds(long s)   { ttlSeconds = s; return this; }
            public Builder edge(String targetId, EdgeType type, float weight) {
                edges.add(new CausalEdge(targetId, type, weight)); return this;
            }
            public Vector build() { return new Vector(this); }
        }
    }

    // ---- CollectionConfig --------------------------------------
    public static final class CollectionConfig {
        private final String name;
        private final int    dim;
        private final String indexType;
        private final int    nShards;
        private final int    nReplicas;
        private final long   halfLifeMs;
        private final float  temporalBlend;

        private CollectionConfig(Builder b) {
            this.name          = b.name;
            this.dim           = b.dim;
            this.indexType     = b.indexType;
            this.nShards       = b.nShards;
            this.nReplicas     = b.nReplicas;
            this.halfLifeMs    = b.halfLifeMs;
            this.temporalBlend = b.temporalBlend;
        }

        public static Builder builder(String name) { return new Builder(name); }

        public String getName()          { return name; }
        public int    getDim()           { return dim; }
        public String getIndexType()     { return indexType; }
        public int    getNShards()       { return nShards; }
        public int    getNReplicas()     { return nReplicas; }
        public long   getHalfLifeMs()   { return halfLifeMs; }
        public float  getTemporalBlend() { return temporalBlend; }

        public static final class Builder {
            private final String name;
            private int    dim          = 768;
            private String indexType    = "tvindex";
            private int    nShards      = 4;
            private int    nReplicas    = 1;
            private long   halfLifeMs   = HalfLife.ONE_MONTH;
            private float  temporalBlend= 0.3f;

            Builder(String name) { this.name = name; }
            public Builder dim(int d)              { dim = d; return this; }
            public Builder indexType(String t)     { indexType = t; return this; }
            public Builder nShards(int n)          { nShards = n; return this; }
            public Builder nReplicas(int n)        { nReplicas = n; return this; }
            public Builder halfLifeMs(long ms)     { halfLifeMs = ms; return this; }
            public Builder temporalBlend(float b)  { temporalBlend = b; return this; }
            public CollectionConfig build() { return new CollectionConfig(this); }
        }
    }

    // ---- SearchRequest -----------------------------------------
    public static final class SearchRequest {
        private final float[]   vector;
        private final int       topK;
        private final float     temporalBlend;
        private final QueryMode mode;
        private final int       causalHops;
        private final String    filter;
        private final boolean   includeTrace;
        private final boolean   includeStalenessWarnings;
        private final Device    device;

        private SearchRequest(Builder b) {
            this.vector                   = b.vector;
            this.topK                     = b.topK;
            this.temporalBlend            = b.temporalBlend;
            this.mode                     = b.mode;
            this.causalHops               = b.causalHops;
            this.filter                   = b.filter;
            this.includeTrace             = b.includeTrace;
            this.includeStalenessWarnings = b.includeStalenessWarnings;
            this.device                   = b.device;
        }

        public static Builder builder(float[] vector) { return new Builder(vector); }

        public float[]   getVector()                  { return vector; }
        public int       getTopK()                    { return topK; }
        public float     getTemporalBlend()           { return temporalBlend; }
        public QueryMode getMode()                    { return mode; }
        public int       getCausalHops()              { return causalHops; }
        public String    getFilter()                  { return filter; }
        public boolean   isIncludeTrace()             { return includeTrace; }
        public boolean   isIncludeStalenessWarnings() { return includeStalenessWarnings; }
        public Device    getDevice()                  { return device; }

        public static final class Builder {
            private final float[]   vector;
            private int       topK                     = 10;
            private float     temporalBlend            = 0.3f;
            private QueryMode mode                     = QueryMode.VECTOR_ONLY;
            private int       causalHops               = 1;
            private String    filter                   = "";
            private boolean   includeTrace             = false;
            private boolean   includeStalenessWarnings = true;
            private Device    device                   = Device.AUTO;

            Builder(float[] vector) { this.vector = vector; }
            public Builder topK(int k)                     { topK = k; return this; }
            public Builder temporalBlend(float b)          { temporalBlend = b; return this; }
            public Builder mode(QueryMode m)               { mode = m; return this; }
            public Builder causalHops(int h)               { causalHops = h; return this; }
            public Builder filter(String f)                { filter = f; return this; }
            public Builder includeTrace(boolean t)         { includeTrace = t; return this; }
            public Builder includeStalenessWarnings(boolean w) { includeStalenessWarnings = w; return this; }
            public Builder device(Device d)                { device = d; return this; }
            public SearchRequest build() { return new SearchRequest(this); }
        }
    }

    // ---- SearchHit ---------------------------------------------
    public static final class SearchHit {
        private final String              id;
        private final float               score;
        private final float               vectorScore;
        private final float               temporalScore;
        private final Map<String, String> payload;
        private final long                createdAt;
        private final boolean             stalenessWarning;
        private final String              stalenessReason;
        private final List<String>        causalNeighbors;
        private final List<String>        contradictedBy;

        SearchHit(io.tidevec.TideVecProto.SearchResult r) {
            this.id               = r.getId();
            this.score            = r.getScore();
            this.vectorScore      = r.getVectorScore();
            this.temporalScore    = r.getTemporalScore();
            this.payload          = Collections.unmodifiableMap(r.getPayloadMap());
            this.createdAt        = r.getCreatedAt();
            this.stalenessWarning = r.getStalenessWarning();
            this.stalenessReason  = r.getStalenessReason();
            this.causalNeighbors  = Collections.unmodifiableList(r.getCausalNeighborsList());
            this.contradictedBy   = Collections.unmodifiableList(r.getContradictedByList());
        }

        public String              getId()               { return id; }
        public float               getScore()            { return score; }
        public float               getVectorScore()      { return vectorScore; }
        public float               getTemporalScore()    { return temporalScore; }
        public Map<String, String> getPayload()          { return payload; }
        public long                getCreatedAt()        { return createdAt; }
        public boolean             isStalenessWarning()  { return stalenessWarning; }
        public String              getStalenessReason()  { return stalenessReason; }
        public List<String>        getCausalNeighbors()  { return causalNeighbors; }
        public List<String>        getContradictedBy()   { return contradictedBy; }

        @Override
        public String toString() {
            return String.format("SearchHit{id='%s', score=%.4f, temporal=%.3f}",
                id, score, temporalScore);
        }
    }

    // ---- SearchResponse ----------------------------------------
    public static final class SearchResponse {
        private final List<SearchHit> hits;
        private final long            count;
        private final double          latencyMs;
        private final String          queryId;
        private final String          strategy;

        SearchResponse(io.tidevec.TideVecProto.SearchResponse r) {
            this.hits      = r.getResultsList().stream()
                              .map(SearchHit::new).collect(Collectors.toList());
            this.count     = r.getCount();
            this.latencyMs = r.getTrace() != null ? r.getTrace().getLatencyMs() : 0;
            this.queryId   = r.getTrace() != null ? r.getTrace().getQueryId()   : "";
            this.strategy  = r.getTrace() != null ? r.getTrace().getStrategy()  : "";
        }

        public List<SearchHit> getHits()      { return hits; }
        public long            getCount()     { return count; }
        public double          getLatencyMs() { return latencyMs; }
        public String          getQueryId()   { return queryId; }
        public String          getStrategy()  { return strategy; }
        public SearchHit       first()        { return hits.isEmpty() ? null : hits.get(0); }
    }

    // ================================================================
    // Client builder
    // ================================================================
    public static final class Builder {
        private final String host;
        private final int    port;
        private String  apiKey  = "";
        private long    timeout = 30_000L;
        private boolean tls     = false;

        Builder(String host, int port) { this.host = host; this.port = port; }
        public Builder apiKey(String k)   { apiKey = k; return this; }
        public Builder timeout(long ms)   { timeout = ms; return this; }
        public Builder tls(boolean t)     { tls = t; return this; }
        public TideVecClient build()     { return new TideVecClient(this); }
    }

    public static Builder builder(String host, int port) {
        return new Builder(host, port);
    }

    // ================================================================
    // Client implementation
    // ================================================================
    private final ManagedChannel          channel;
    private final TideVecGrpc.TideVecBlockingStub  blockingStub;
    private final TideVecGrpc.TideVecFutureStub    futureStub;
    private final String                  apiKey;
    private final long                    timeoutMs;

    private TideVecClient(Builder b) {
        ManagedChannelBuilder<?> cb = ManagedChannelBuilder
            .forAddress(b.host, b.port)
            .maxInboundMessageSize(256 * 1024 * 1024);
        if (!b.tls) cb = cb.usePlaintext();
        this.channel     = cb.build();
        this.apiKey      = b.apiKey;
        this.timeoutMs   = b.timeout;

        Metadata.Key<String> KEY = Metadata.Key.of("x-api-key", Metadata.ASCII_STRING_MARSHALLER);
        Metadata headers = new Metadata();
        if (!apiKey.isEmpty()) headers.put(KEY, apiKey);

        this.blockingStub = MetadataUtils.attachHeaders(
            TideVecGrpc.newBlockingStub(channel), headers);
        this.futureStub   = MetadataUtils.attachHeaders(
            TideVecGrpc.newFutureStub(channel), headers);
    }

    private <T> T call(java.util.function.Supplier<T> fn) {
        return fn.get();
    }

    // ---- Health ------------------------------------------------
    public boolean ping() {
        try {
            blockingStub.withDeadlineAfter(5, TimeUnit.SECONDS)
                .health(TideVecProto.HealthRequest.newBuilder().build());
            return true;
        } catch (Exception e) { return false; }
    }

    // ---- Collections -------------------------------------------
    public void createCollection(CollectionConfig cfg) {
        blockingStub.withDeadlineAfter(timeoutMs, TimeUnit.MILLISECONDS)
            .createCollection(TideVecProto.CreateCollectionRequest.newBuilder()
                .setName(cfg.getName())
                .setDim(cfg.getDim())
                .setIndexType(cfg.getIndexType())
                .setNShards(cfg.getNShards())
                .setNReplicas(cfg.getNReplicas())
                .setTemporal(TideVecProto.TemporalConfig.newBuilder()
                    .setHalfLifeMs(cfg.getHalfLifeMs())
                    .setTemporalBlend(cfg.getTemporalBlend())
                    .build())
                .build());
    }

    public void dropCollection(String name) {
        blockingStub.withDeadlineAfter(timeoutMs, TimeUnit.MILLISECONDS)
            .dropCollection(TideVecProto.DropCollectionRequest.newBuilder()
                .setName(name).build());
    }

    // ---- Vectors -----------------------------------------------
    public int upsert(String collection, List<Vector> vectors) {
        List<TideVecProto.Vector> pbVecs = vectors.stream().map(v -> {
            TideVecProto.Vector.Builder vb = TideVecProto.Vector.newBuilder()
                .setId(v.getId())
                .addAllEmbedding(toFloatList(v.getEmbedding()))
                .putAllPayload(v.getPayload())
                .setTtlSeconds(v.getTtlSeconds());
            for (CausalEdge e : v.getEdges()) {
                vb.addEdges(TideVecProto.CausalEdge.newBuilder()
                    .setTargetId(e.getTargetId())
                    .setWeight(e.getWeight())
                    .build());
            }
            return vb.build();
        }).collect(Collectors.toList());

        TideVecProto.UpsertResponse resp = blockingStub
            .withDeadlineAfter(timeoutMs, TimeUnit.MILLISECONDS)
            .upsert(TideVecProto.UpsertRequest.newBuilder()
                .setCollection(collection)
                .addAllVectors(pbVecs)
                .build());
        return (int) resp.getInserted();
    }

    public int delete(String collection, List<String> ids) {
        TideVecProto.DeleteResponse resp = blockingStub
            .withDeadlineAfter(timeoutMs, TimeUnit.MILLISECONDS)
            .delete(TideVecProto.DeleteRequest.newBuilder()
                .setCollection(collection)
                .addAllIds(ids)
                .build());
        return (int) resp.getDeleted();
    }

    // ---- Search ------------------------------------------------
    public SearchResponse search(String collection, SearchRequest req) {
        TideVecProto.SearchOptions.Builder opts = TideVecProto.SearchOptions.newBuilder()
            .setTopK(req.getTopK())
            .setTemporalBlend(req.getTemporalBlend())
            .setFilter(req.getFilter())
            .setIncludeTrace(req.isIncludeTrace())
            .setIncludeStalenessWarnings(req.isIncludeStalenessWarnings());

        TideVecProto.SearchResponse resp = blockingStub
            .withDeadlineAfter(timeoutMs, TimeUnit.MILLISECONDS)
            .search(TideVecProto.SearchRequest.newBuilder()
                .setCollection(collection)
                .addAllVector(toFloatList(req.getVector()))
                .setOptions(opts.build())
                .build());
        return new SearchResponse(resp);
    }

    /** Async search — returns CompletableFuture. */
    public CompletableFuture<SearchResponse> searchAsync(String collection, SearchRequest req) {
        CompletableFuture<SearchResponse> future = new CompletableFuture<>();
        TideVecProto.SearchOptions opts = TideVecProto.SearchOptions.newBuilder()
            .setTopK(req.getTopK())
            .setTemporalBlend(req.getTemporalBlend())
            .build();

        TideVecGrpc.newStub(channel).search(
            TideVecProto.SearchRequest.newBuilder()
                .setCollection(collection)
                .addAllVector(toFloatList(req.getVector()))
                .setOptions(opts).build(),
            new StreamObserver<TideVecProto.SearchResponse>() {
                @Override public void onNext(TideVecProto.SearchResponse r) {
                    future.complete(new SearchResponse(r));
                }
                @Override public void onError(Throwable t) { future.completeExceptionally(t); }
                @Override public void onCompleted() {}
            }
        );
        return future;
    }

    // ---- Graph -------------------------------------------------
    public int addEdges(String collection, List<io.tidevec.Edge> edges) {
        List<TideVecProto.Edge> pbEdges = edges.stream()
            .map(e -> TideVecProto.Edge.newBuilder()
                .setSrc(e.getSrc()).setTgt(e.getTgt())
                .setWeight(e.getWeight()).build())
            .collect(Collectors.toList());

        TideVecProto.AddEdgesResponse resp = blockingStub
            .withDeadlineAfter(timeoutMs, TimeUnit.MILLISECONDS)
            .addEdges(TideVecProto.AddEdgesRequest.newBuilder()
                .setCollection(collection)
                .addAllEdges(pbEdges).build());
        return (int) resp.getAdded();
    }

    // ---- AutoCloseable -----------------------------------------
    @Override
    public void close() {
        try {
            channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    // ---- Helpers -----------------------------------------------
    private static List<Float> toFloatList(float[] arr) {
        List<Float> list = new ArrayList<>(arr.length);
        for (float f : arr) list.add(f);
        return list;
    }
}
