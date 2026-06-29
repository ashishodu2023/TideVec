package io.tidevec;

/** A causal edge between two vectors. */
public final class Edge {
    private final String                       src;
    private final String                       tgt;
    private final TideVecClient.EdgeType      type;
    private final float                        weight;

    public Edge(String src, String tgt, TideVecClient.EdgeType type, float weight) {
        this.src    = src;
        this.tgt    = tgt;
        this.type   = type;
        this.weight = weight;
    }

    public String                  getSrc()    { return src; }
    public String                  getTgt()    { return tgt; }
    public TideVecClient.EdgeType getType()   { return type; }
    public float                   getWeight() { return weight; }
}
