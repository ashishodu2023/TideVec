// Package tidevec provides a Go client for TideVec,
// the temporally-aware causal vector database.
//
// Quick start:
//
//	db, err := tidevec.New("localhost:6399")
//	if err != nil { log.Fatal(err) }
//	defer db.Close()
//
//	err = db.CreateCollection(ctx, tidevec.CollectionConfig{
//	    Name:          "docs",
//	    Dim:           768,
//	    HalfLifeMs:    tidevec.HalfLifeOneWeek,
//	    TemporalBlend: 0.3,
//	})
//
//	err = db.Upsert(ctx, "docs", []tidevec.Vector{{
//	    ID:        "v1",
//	    Embedding: []float32{0.1, 0.2, ...},
//	    Payload:   map[string]string{"src": "wiki"},
//	}})
//
//	resp, err := db.Search(ctx, "docs", tidevec.SearchRequest{
//	    Vector:        []float32{0.1, 0.2, ...},
//	    TopK:          10,
//	    TemporalBlend: 0.3,
//	})
package tidevec

import (
	"context"
	"fmt"
	"time"

	pb "github.com/ashishodu2023/TideVec/sdk/go/tidevec"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

// ================================================================
// Half-life presets (milliseconds)
// ================================================================
const (
	HalfLifeOneHour  int64 = 3_600_000
	HalfLifeOneDay   int64 = 86_400_000
	HalfLifeOneWeek  int64 = 604_800_000
	HalfLifeOneMonth int64 = 2_592_000_000
	HalfLifeOneYear  int64 = 31_536_000_000

	// Semantic aliases
	HalfLifeAgentSession  = HalfLifeOneHour
	HalfLifeNewsFeed      = HalfLifeOneDay
	HalfLifeSupportTicket = HalfLifeOneMonth
	HalfLifeDocumentStore = HalfLifeOneYear
)

// ================================================================
// Domain types
// ================================================================

// EdgeType defines the semantic relationship between two vectors.
type EdgeType string

const (
	EdgeCauses      EdgeType = "CAUSES"
	EdgeContradicts EdgeType = "CONTRADICTS"
	EdgeUpdates     EdgeType = "UPDATES"
	EdgeRelatedTo   EdgeType = "RELATED_TO"
	EdgeEntityOf    EdgeType = "ENTITY_OF"
	EdgeSupports    EdgeType = "SUPPORTS"
)

// QueryMode controls how results are expanded.
type QueryMode string

const (
	ModeVectorOnly         QueryMode = "vector_only"
	ModeCausalExpand       QueryMode = "causal_expand"
	ModeContradictionCheck QueryMode = "contradiction_check"
	ModeEntityResolve      QueryMode = "entity_resolve"
)

// Device hint for accelerator routing.
type Device string

const (
	DeviceAuto Device = "auto"
	DeviceCPU  Device = "cpu"
	DeviceGPU  Device = "gpu"
	DeviceTPU  Device = "tpu"
)

// CausalEdge links two vectors with a typed relationship.
type CausalEdge struct {
	TargetID string
	Type     EdgeType
	Weight   float32
}

// Vector is the core unit stored in TideVec.
type Vector struct {
	ID          string
	Embedding   []float32
	Payload     map[string]string
	CreatedAt   int64  // ms since epoch; 0 = server-assigned
	ValidUntil  int64  // ms since epoch; 0 = no expiry
	TTLSeconds  int64
	Edges       []CausalEdge
}

// CollectionConfig holds parameters for CreateCollection.
type CollectionConfig struct {
	Name           string
	Dim            int
	IndexType      string  // "tvindex" | "flat"
	Metric         string  // "cosine" | "l2" | "dot"
	NShards        int
	NReplicas      int
	WriteQuorum    int
	HalfLifeMs     int64
	TemporalBlend  float32
}

// CollectionInfo holds collection metadata.
type CollectionInfo struct {
	Name      string
	NVectors  uint64
	NShards   uint32
	Dim       uint32
	IndexType string
	Metric    string
}

// SearchRequest specifies a vector search.
type SearchRequest struct {
	Vector                  []float32
	TopK                    int
	TemporalBlend           float32
	Mode                    QueryMode
	CausalHops              int
	Filter                  string
	Metric                  string
	IncludeTrace            bool
	IncludeStalenessWarnings bool
	Device                  Device
}

// SearchHit is one result from a vector search.
type SearchHit struct {
	ID               string
	Score            float32
	VectorScore      float32
	TemporalScore    float32
	Payload          map[string]string
	CreatedAt        int64
	StalenessWarning bool
	StalenessReason  string
	CausalNeighbors  []string
	ContradictedBy   []string
}

// SearchResponse wraps results from a search call.
type SearchResponse struct {
	Hits      []SearchHit
	Count     uint64
	LatencyMs float64
	QueryID   string
	Strategy  string
}

// Edge defines a causal relationship for AddEdges.
type Edge struct {
	Src    string
	Tgt    string
	Type   EdgeType
	Weight float32
}

// HealthResponse holds server health info.
type HealthResponse struct {
	Status       string
	Version      string
	Collections  uint64
	TimestampMs  uint64
	GPUAvailable bool
	TPUAvailable bool
}

// ================================================================
// Client
// ================================================================

// Client is a TideVec gRPC client.
type Client struct {
	conn    *grpc.ClientConn
	stub    pb.TideVecClient
	apiKey  string
	timeout time.Duration
}

// Option configures a Client.
type Option func(*Client)

// WithAPIKey sets the API key sent with every request.
func WithAPIKey(key string) Option {
	return func(c *Client) { c.apiKey = key }
}

// WithTimeout sets the default RPC timeout.
func WithTimeout(d time.Duration) Option {
	return func(c *Client) { c.timeout = d }
}

// New creates a new TideVec client.
func New(addr string, opts ...Option) (*Client, error) {
	conn, err := grpc.Dial(addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(256*1024*1024),
			grpc.MaxCallSendMsgSize(256*1024*1024),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("tidevec: dial %s: %w", addr, err)
	}
	c := &Client{
		conn:    conn,
		stub:    pb.NewTideVecClient(conn),
		timeout: 30 * time.Second,
	}
	for _, o := range opts {
		o(c)
	}
	return c, nil
}

// Close releases the gRPC connection.
func (c *Client) Close() error { return c.conn.Close() }

func (c *Client) ctx(parent context.Context) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithTimeout(parent, c.timeout)
	if c.apiKey != "" {
		ctx = metadata.AppendToOutgoingContext(ctx, "x-api-key", c.apiKey)
	}
	return ctx, cancel
}

// ---- Health ----------------------------------------------------

// Health checks server health.
func (c *Client) Health(parent context.Context) (*HealthResponse, error) {
	ctx, cancel := c.ctx(parent)
	defer cancel()
	resp, err := c.stub.Health(ctx, &pb.HealthRequest{})
	if err != nil {
		return nil, fmt.Errorf("tidevec: health: %w", err)
	}
	return &HealthResponse{
		Status:       resp.Status,
		Version:      resp.Version,
		Collections:  resp.Collections,
		TimestampMs:  resp.TimestampMs,
		GPUAvailable: resp.GpuAvailable,
		TPUAvailable: resp.TpuAvailable,
	}, nil
}

// Ping returns true if server is reachable.
func (c *Client) Ping(parent context.Context) bool {
	_, err := c.Health(parent)
	return err == nil
}

// ---- Collections -----------------------------------------------

// CreateCollection creates a new vector collection.
func (c *Client) CreateCollection(parent context.Context, cfg CollectionConfig) error {
	ctx, cancel := c.ctx(parent)
	defer cancel()

	if cfg.IndexType == "" {
		cfg.IndexType = "tvindex"
	}
	if cfg.Metric == "" {
		cfg.Metric = "cosine"
	}
	if cfg.NShards == 0 {
		cfg.NShards = 4
	}
	if cfg.NReplicas == 0 {
		cfg.NReplicas = 1
	}
	if cfg.WriteQuorum == 0 {
		cfg.WriteQuorum = 1
	}
	if cfg.HalfLifeMs == 0 {
		cfg.HalfLifeMs = HalfLifeOneMonth
	}
	if cfg.TemporalBlend == 0 {
		cfg.TemporalBlend = 0.3
	}

	_, err := c.stub.CreateCollection(ctx, &pb.CreateCollectionRequest{
		Name:        cfg.Name,
		Dim:         uint32(cfg.Dim),
		IndexType:   cfg.IndexType,
		NShards:     uint32(cfg.NShards),
		NReplicas:   uint32(cfg.NReplicas),
		WriteQuorum: uint32(cfg.WriteQuorum),
		Temporal: &pb.TemporalConfig{
			HalfLifeMs:    cfg.HalfLifeMs,
			TemporalBlend: cfg.TemporalBlend,
		},
	})
	if err != nil {
		return fmt.Errorf("tidevec: create_collection %s: %w", cfg.Name, err)
	}
	return nil
}

// DropCollection deletes a collection and all its vectors.
func (c *Client) DropCollection(parent context.Context, name string) error {
	ctx, cancel := c.ctx(parent)
	defer cancel()
	_, err := c.stub.DropCollection(ctx, &pb.DropCollectionRequest{Name: name})
	return err
}

// GetCollection returns collection metadata.
func (c *Client) GetCollection(parent context.Context, name string) (*CollectionInfo, error) {
	ctx, cancel := c.ctx(parent)
	defer cancel()
	resp, err := c.stub.GetCollection(ctx, &pb.GetCollectionRequest{Name: name})
	if err != nil {
		return nil, fmt.Errorf("tidevec: get_collection %s: %w", name, err)
	}
	return &CollectionInfo{
		Name:     resp.Info.Name,
		NVectors: resp.Info.NVectors,
		NShards:  resp.Info.NShards,
		Dim:      resp.Info.Dim,
	}, nil
}

// ListCollections returns all collections.
func (c *Client) ListCollections(parent context.Context) ([]CollectionInfo, error) {
	ctx, cancel := c.ctx(parent)
	defer cancel()
	resp, err := c.stub.ListCollections(ctx, &pb.ListCollectionsRequest{})
	if err != nil {
		return nil, err
	}
	out := make([]CollectionInfo, len(resp.Collections))
	for i, col := range resp.Collections {
		out[i] = CollectionInfo{
			Name: col.Name, NVectors: col.NVectors,
			NShards: col.NShards, Dim: col.Dim,
		}
	}
	return out, nil
}

// SetTemporal updates the temporal decay config for a collection.
func (c *Client) SetTemporal(parent context.Context, name string,
	halfLifeMs int64, blend float32) error {
	ctx, cancel := c.ctx(parent)
	defer cancel()
	_, err := c.stub.UpdateTemporal(ctx, &pb.UpdateTemporalRequest{
		Name: name,
		Config: &pb.TemporalConfig{
			HalfLifeMs: halfLifeMs, TemporalBlend: blend,
		},
	})
	return err
}

// ---- Vectors ---------------------------------------------------

func edgeTypeToProto(t EdgeType) pb.EdgeType {
	switch t {
	case EdgeCauses:      return pb.EdgeType_CAUSES
	case EdgeContradicts: return pb.EdgeType_CONTRADICTS
	case EdgeUpdates:     return pb.EdgeType_UPDATES
	case EdgeRelatedTo:   return pb.EdgeType_RELATED_TO
	case EdgeEntityOf:    return pb.EdgeType_ENTITY_OF
	case EdgeSupports:    return pb.EdgeType_SUPPORTS
	default:              return pb.EdgeType_RELATED_TO
	}
}

// Upsert inserts or updates vectors in a collection.
func (c *Client) Upsert(parent context.Context, collection string, vectors []Vector) error {
	ctx, cancel := c.ctx(parent)
	defer cancel()

	pbVecs := make([]*pb.Vector, len(vectors))
	for i, v := range vectors {
		edges := make([]*pb.CausalEdge, len(v.Edges))
		for j, e := range v.Edges {
			edges[j] = &pb.CausalEdge{
				TargetId: e.TargetID,
				Type:     edgeTypeToProto(e.Type),
				Weight:   e.Weight,
			}
		}
		pbVecs[i] = &pb.Vector{
			Id:         v.ID,
			Embedding:  v.Embedding,
			Payload:    v.Payload,
			TtlSeconds: v.TTLSeconds,
			Edges:      edges,
		}
	}
	_, err := c.stub.Upsert(ctx, &pb.UpsertRequest{
		Collection: collection,
		Vectors:    pbVecs,
	})
	return err
}

// Delete removes vectors by ID.
func (c *Client) Delete(parent context.Context, collection string, ids []string) (uint64, error) {
	ctx, cancel := c.ctx(parent)
	defer cancel()
	resp, err := c.stub.Delete(ctx, &pb.DeleteRequest{
		Collection: collection, Ids: ids,
	})
	if err != nil {
		return 0, err
	}
	return resp.Deleted, nil
}

// ---- Search ----------------------------------------------------

func modeToProto(m QueryMode) pb.QueryMode {
	switch m {
	case ModeCausalExpand:       return pb.QueryMode_CAUSAL_EXPAND
	case ModeContradictionCheck: return pb.QueryMode_CONTRADICTION_CHECK
	case ModeEntityResolve:      return pb.QueryMode_ENTITY_RESOLVE
	default:                     return pb.QueryMode_VECTOR_ONLY
	}
}

func deviceToProto(d Device) pb.Device {
	switch d {
	case DeviceCPU: return pb.Device_CPU
	case DeviceGPU: return pb.Device_GPU
	case DeviceTPU: return pb.Device_TPU
	default:        return pb.Device_AUTO
	}
}

func hitFromProto(r *pb.SearchResult) SearchHit {
	return SearchHit{
		ID: r.Id, Score: r.Score,
		VectorScore: r.VectorScore, TemporalScore: r.TemporalScore,
		Payload: r.Payload, CreatedAt: r.CreatedAt,
		StalenessWarning: r.StalenessWarning,
		StalenessReason:  r.StalenessReason,
		CausalNeighbors:  r.CausalNeighbors,
		ContradictedBy:   r.ContradictedBy,
	}
}

// Search finds nearest neighbours with temporal scoring.
func (c *Client) Search(parent context.Context, collection string,
	req SearchRequest) (*SearchResponse, error) {

	ctx, cancel := c.ctx(parent)
	defer cancel()

	if req.TopK == 0 {
		req.TopK = 10
	}

	opts := &pb.SearchOptions{
		TopK:                     uint32(req.TopK),
		TemporalBlend:            req.TemporalBlend,
		Mode:                     modeToProto(req.Mode),
		CausalHops:               uint32(req.CausalHops),
		Filter:                   req.Filter,
		IncludeTrace:             req.IncludeTrace,
		IncludeStalenessWarnings: req.IncludeStalenessWarnings,
		DeviceHint:               deviceToProto(req.Device),
	}

	resp, err := c.stub.Search(ctx, &pb.SearchRequest{
		Collection: collection,
		Vector:     req.Vector,
		Options:    opts,
	})
	if err != nil {
		return nil, fmt.Errorf("tidevec: search: %w", err)
	}

	hits := make([]SearchHit, len(resp.Results))
	for i, r := range resp.Results {
		hits[i] = hitFromProto(r)
	}

	out := &SearchResponse{
		Hits:  hits,
		Count: resp.Count,
	}
	if req.IncludeTrace && resp.Trace != nil {
		out.LatencyMs = resp.Trace.LatencyMs
		out.QueryID   = resp.Trace.QueryId
		out.Strategy  = resp.Trace.Strategy
	}
	return out, nil
}

// BatchSearch sends multiple queries in one GPU/TPU call.
func (c *Client) BatchSearch(parent context.Context, collection string,
	queries [][]float32, topK int, blend float32, device Device) ([]SearchResponse, error) {

	ctx, cancel := c.ctx(parent)
	defer cancel()

	reqs := make([]*pb.SearchRequest, len(queries))
	for i, q := range queries {
		reqs[i] = &pb.SearchRequest{
			Collection: collection,
			Vector:     q,
			Options: &pb.SearchOptions{
				TopK:          uint32(topK),
				TemporalBlend: blend,
				DeviceHint:    deviceToProto(device),
			},
		}
	}
	resp, err := c.stub.BatchSearch(ctx, &pb.BatchSearchRequest{
		Collection: collection, Queries: reqs,
	})
	if err != nil {
		return nil, err
	}
	out := make([]SearchResponse, len(resp.Responses))
	for i, sr := range resp.Responses {
		hits := make([]SearchHit, len(sr.Results))
		for j, r := range sr.Results {
			hits[j] = hitFromProto(r)
		}
		out[i] = SearchResponse{Hits: hits, Count: sr.Count}
	}
	return out, nil
}

// ---- Graph -----------------------------------------------------

// AddEdges creates causal relationships between vectors.
func (c *Client) AddEdges(parent context.Context, collection string, edges []Edge) (uint64, error) {
	ctx, cancel := c.ctx(parent)
	defer cancel()
	pbEdges := make([]*pb.Edge, len(edges))
	for i, e := range edges {
		pbEdges[i] = &pb.Edge{
			Src: e.Src, Tgt: e.Tgt,
			Type: edgeTypeToProto(e.Type), Weight: e.Weight,
		}
	}
	resp, err := c.stub.AddEdges(ctx, &pb.AddEdgesRequest{
		Collection: collection, Edges: pbEdges,
	})
	if err != nil {
		return 0, err
	}
	return resp.Added, nil
}
