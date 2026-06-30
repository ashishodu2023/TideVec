package io.tidevec;

/**
 * A causal edge between two vectors.
 *
 * Mirrors {@link TideVecClient.CausalEdge} as a standalone, top-level
 * type for callers who prefer importing {@code Edge} directly instead
 * of the nested record.
 *
 * Valid {@code type} values match the server's EdgeType enum:
 * CAUSES, CONTRADICTS, UPDATES, RELATED_TO, ENTITY_OF, SUPPORTS.
 */
public final class Edge {
    private final String src;
    private final String tgt;
    private final String type;
    private final float  weight;

    public Edge(String src, String tgt, String type, float weight) {
        this.src    = src;
        this.tgt    = tgt;
        this.type   = type;
        this.weight = weight;
    }

    public static Edge causes(String src, String tgt)      { return new Edge(src, tgt, "CAUSES", 1.0f); }
    public static Edge updates(String src, String tgt)     { return new Edge(src, tgt, "UPDATES", 1.0f); }
    public static Edge contradicts(String src, String tgt) { return new Edge(src, tgt, "CONTRADICTS", 1.0f); }

    public String getSrc()    { return src; }
    public String getTgt()    { return tgt; }
    public String getType()   { return type; }
    public float  getWeight() { return weight; }

    /** Convert to the wire-format CausalEdge used by TideVecClient.upsert(). */
    public TideVecClient.CausalEdge toCausalEdge() {
        return new TideVecClient.CausalEdge(tgt, type, weight);
    }
}