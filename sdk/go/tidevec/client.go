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
//	    Supersession:  &tidevec.SupersessionConfig{MatchField: "doc_key"},
//	})
//
//	res, err := db.Search(ctx, "docs", queryVec, tidevec.SearchOptions{TopK: 10})
package tidevec

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"
)

// Half-life presets in milliseconds.
const (
	HalfLifeOneHour  int64 = 3_600_000
	HalfLifeOneDay   int64 = 86_400_000
	HalfLifeOneWeek  int64 = 604_800_000
	HalfLifeOneMonth int64 = 2_592_000_000
	HalfLifeOneYear  int64 = 31_536_000_000
)

// EdgeType represents a causal relationship type.
type EdgeType string

const (
	EdgeCauses       EdgeType = "CAUSES"
	EdgeContradicts  EdgeType = "CONTRADICTS"
	EdgeUpdates      EdgeType = "UPDATES"
	EdgeRelatedTo    EdgeType = "RELATED_TO"
	EdgeEntityOf     EdgeType = "ENTITY_OF"
	EdgeSupports     EdgeType = "SUPPORTS"
	EdgeSupersededBy EdgeType = "SUPERSEDED_BY"
)

// CausalEdge links a vector to a target with a typed relationship
// (used inline in Vector.Edges).
type CausalEdge struct {
	TargetID string   `json:"target_id"`
	Type     EdgeType `json:"type"`
	Weight   float32  `json:"weight"`
}

// Edge is a standalone edge for AddEdges.
type Edge struct {
	Src    string   `json:"src"`
	Tgt    string   `json:"tgt"`
	Type   EdgeType `json:"type"`
	Weight float32  `json:"weight"`
}

// Vector is a record stored in a TideVec collection.
type Vector struct {
	ID          string            `json:"id"`
	Embedding   []float32         `json:"embedding"`
	Payload     map[string]string `json:"payload,omitempty"`
	Edges       []CausalEdge      `json:"edges,omitempty"`
	TimestampMs *int64            `json:"timestamp_ms,omitempty"`
}

// CollectionConfig holds settings for a new collection. Zero-valued
// temporal fields are omitted so the server defaults apply.
type CollectionConfig struct {
	Name               string
	Dim                int
	Metric             string
	HalfLifeMs         int64
	TemporalBlend      float32
	NShards            int
	NReplicas          int
	IndexType          string
	StalenessThreshold float32
	// Supersession enables automatic newer-version detection on upsert.
	Supersession *SupersessionConfig
	// TextField is the payload field indexed for BM25 keyword / hybrid
	// search (SearchOptions.Text). Empty disables keyword search.
	TextField string
	// DisablePayloadIndex turns off the per-key payload index used to
	// plan selective filters (saves memory on write-heavy collections).
	DisablePayloadIndex bool
	// CompactionRatio: indexes are rebuilt in the background once
	// deleted/re-embedded slots reach this fraction of the live vectors.
	// 0 means the server default (1.0). See Client.Compact.
	CompactionRatio float64
}

// Filter is a JSON filter object:
//
//	Filter{"lang": "en"}                                 equality
//	Filter{"lang": Filter{"$in": []string{"en", "de"}}}  $eq $ne $in $nin
//	Filter{"year": Filter{"$gte": 2020, "$lt": 2025}}    $gt $gte $lt $lte (numbers or ISO-8601 dates)
//	Filter{"$or": []Filter{{...}, {...}}}                $and $or $not, {"k": {"$exists": true}}
//
// Several keys in one object are ANDed; "created_at" and "_id" are
// always available.
type Filter = map[string]any

// HybridOptions controls how vector and keyword rankings are fused.
type HybridOptions struct {
	Fusion string `json:"fusion,omitempty"` // "rrf" (default) | "weighted"
	// Alpha weights the vector side (0–1); zero means the default 0.5.
	// For keyword-only search pass a nil embedding instead.
	Alpha float32 `json:"alpha,omitempty"`
	RRFK  int     `json:"rrf_k,omitempty"`
}

// SupersessionConfig controls automatic newer-version detection.
//
// SimilarityThreshold is compared against the engine's vector score,
// which for cosine is (1+cos)/2: 0.92 corresponds to cos ≈ 0.84.
type SupersessionConfig struct {
	SimilarityThreshold float32 `json:"similarity_threshold,omitempty"`
	MatchField          string  `json:"match_field,omitempty"`
	Candidates          int     `json:"candidates,omitempty"`
}

// SearchOptions controls how a search query is executed.
type SearchOptions struct {
	TopK int
	// TemporalBlend overrides the collection's blend for this query;
	// nil uses the collection setting. Use Blend(0) for semantic-only.
	TemporalBlend     *float32
	Mode              string // vector_only | causal_expand | contradiction_check
	CausalHops        int
	Filter            string // "key=value"
	EfSearch          int
	IncludeTrace      bool
	StalenessWarnings bool
	// SupersededPenalty scales down results that have a newer version (0–1).
	SupersededPenalty float32
	// Where is a rich filter; it takes precedence over Filter.
	Where Filter
	// AsOf searches the collection as it was at that time (only vectors
	// created by then, supersession as of then). Zero means now.
	AsOf time.Time
	// Text is a BM25 keyword query over the collection's TextField. With
	// a nil embedding the search is keyword-only; with both the rankings
	// are fused (Hybrid, default RRF).
	Text   string
	Hybrid *HybridOptions
}

// Blend returns a pointer for SearchOptions.TemporalBlend.
func Blend(v float32) *float32 { return &v }

// SearchHit is a single result from a search query.
type SearchHit struct {
	ID               string            `json:"id"`
	Score            float32           `json:"score"`
	SemanticScore    float32           `json:"vector_score"`
	TemporalScore    float32           `json:"temporal_score"`
	CreatedAt        int64             `json:"created_at"`
	Payload          map[string]string `json:"payload"`
	StalenessWarning bool              `json:"staleness_warning"`
	StalenessReason  string            `json:"staleness_reason"`
	CausalNeighbors  []string          `json:"causal_neighbors,omitempty"`
	ContradictedBy   []string          `json:"contradicted_by,omitempty"`
	SupersededBy     []string          `json:"superseded_by,omitempty"`
	TextScore        float32           `json:"text_score,omitempty"`
}

// IsSuperseded reports whether a newer version of this record exists.
func (h SearchHit) IsSuperseded() bool { return len(h.SupersededBy) > 0 }

// StoredEdge is an edge attached to a Record.
type StoredEdge struct {
	TargetID  string   `json:"target_id"`
	Type      EdgeType `json:"type"`
	Weight    float32  `json:"weight"`
	CreatedAt int64    `json:"created_at"`
}

// Record is a stored vector returned by Get / Scroll.
type Record struct {
	ID           string            `json:"id"`
	Embedding    []float32         `json:"embedding,omitempty"`
	Payload      map[string]string `json:"payload,omitempty"`
	CreatedAt    int64             `json:"created_at"`
	ValidFrom    int64             `json:"valid_from"`
	ValidUntil   *int64            `json:"valid_until,omitempty"`
	Edges        []StoredEdge      `json:"edges,omitempty"`
	SupersededBy []string          `json:"superseded_by,omitempty"`
	Supersedes   []string          `json:"supersedes,omitempty"`
}

// ScrollOptions selects a page of records. Pages are in id order; pass
// the previous page's NextCursor as Cursor.
type ScrollOptions struct {
	Limit          int // default 100, max 10000
	Cursor         string
	Filter         string // "key=value"
	Where          Filter
	AsOf           time.Time
	WithVector     bool
	WithoutPayload bool
}

// ScrollPage is one page of records; NextCursor is empty on the last page.
type ScrollPage struct {
	Records    []Record `json:"vectors"`
	NextCursor string   `json:"next_cursor"`
}

// SearchResponse wraps search results.
type SearchResponse struct {
	Hits  []SearchHit `json:"results"`
	Count int         `json:"count"`
}

// ClientConfig holds optional settings for the HTTP client.
type ClientConfig struct {
	APIKey     string
	TLS        bool // use https:// instead of http://
	TimeoutMs  int
	MaxRetries int
}

// DriftStatus reports zero-downtime model migration progress.
type DriftStatus struct {
	Collection   string  `json:"collection"`
	Phase        string  `json:"phase"`
	TotalVectors int     `json:"total_vectors"`
	Migrated     int     `json:"migrated"`
	Skipped      int     `json:"skipped"`
	PctComplete  float64 `json:"pct_complete"`
	Error        string  `json:"error"`
}

// CompactionStatus is a collection's index compaction state; counts are
// summed over shards.
type CompactionStatus struct {
	Started       int     `json:"started"` // shards this Compact call started rebuilding
	Running       bool    `json:"running"`
	RunningShards int     `json:"running_shards"`
	Runs          int     `json:"runs"`
	Tombstones    int     `json:"tombstones"`
	IndexBytes    int64   `json:"index_bytes"`
	LastReclaimed int     `json:"last_reclaimed"`
	LastReplayed  int     `json:"last_replayed"`
	LastMs        float64 `json:"last_ms"`
}

// CollectionInfo describes a collection on the server.
type CollectionInfo struct {
	Name         string              `json:"name"`
	NVectors     int                 `json:"n_vectors"`
	NShards      int                 `json:"n_shards"`
	Dim          int                 `json:"dim"`
	IndexType    string              `json:"index_type"`
	Metric       string              `json:"metric"`
	Backend      string              `json:"backend"`
	TotalWrites  int64               `json:"total_writes"`
	TotalQueries int64               `json:"total_queries"`
	Supersession *SupersessionStatus `json:"supersession,omitempty"`
	TextField    string              `json:"text_field,omitempty"`
	PayloadIndex bool                `json:"payload_index"`
	Aliases      []string            `json:"aliases,omitempty"`
	// Alias is set when the collection was looked up through an alias.
	Alias string `json:"alias,omitempty"`
}

// SupersessionStatus is the supersession config reported by the server.
type SupersessionStatus struct {
	Enabled             bool    `json:"enabled"`
	SimilarityThreshold float32 `json:"similarity_threshold"`
	MatchField          string  `json:"match_field"`
	Candidates          int     `json:"candidates"`
}

// ClusterStatus is the Raft state of one node (experimental cluster mode).
// Mode is "standalone" on a single server and the other fields are zero.
type ClusterStatus struct {
	Mode         string   `json:"mode"`
	NodeID       int      `json:"node_id"`
	Role         string   `json:"role"`
	Term         uint64   `json:"term"`
	LeaderID     int      `json:"leader_id"`
	Leader       string   `json:"leader"`
	CommitIndex  uint64   `json:"commit_index"`
	LastApplied  uint64   `json:"last_applied"`
	LastLogIndex uint64   `json:"last_log_index"`
	Peers        []string `json:"peers"`
}

// APIError is returned for any non-2xx server response.
type APIError struct {
	Status  int
	Message string
	// Leader is set on 421 responses in cluster mode.
	Leader string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("tidevec: server error %d: %s", e.Status, e.Message)
}

// IsNotFound reports whether err is a 404 (unknown collection or route).
func IsNotFound(err error) bool {
	var ae *APIError
	return errors.As(err, &ae) && ae.Status == http.StatusNotFound
}

// Client is a TideVec REST client. It is safe for concurrent use.
type Client struct {
	baseURL    string
	httpClient *http.Client
	apiKey     string
	maxRetries int
}

// New creates a TideVec client connected to the given host:port.
// APIKey falls back to TIDEVEC_API_KEY env var if not set in config.
func New(hostPort string, cfg ...ClientConfig) (*Client, error) {
	timeout := 30 * time.Second
	apiKey := os.Getenv("TIDEVEC_API_KEY")
	useTLS := false
	maxRetries := 3

	if len(cfg) > 0 {
		if cfg[0].TimeoutMs > 0 {
			timeout = time.Duration(cfg[0].TimeoutMs) * time.Millisecond
		}
		if cfg[0].APIKey != "" {
			apiKey = cfg[0].APIKey
		}
		useTLS = cfg[0].TLS
		if cfg[0].MaxRetries > 0 {
			maxRetries = cfg[0].MaxRetries
		}
	}

	scheme := "http"
	if useTLS {
		scheme = "https"
	}
	return &Client{
		baseURL:    scheme + "://" + hostPort,
		httpClient: &http.Client{Timeout: timeout},
		apiKey:     apiKey,
		maxRetries: maxRetries,
	}, nil
}

// Close is a no-op for the HTTP client (provided for interface symmetry).
func (c *Client) Close() {}

// Ping checks whether the server is reachable.
func (c *Client) Ping(ctx context.Context) bool {
	_, err := c.do(ctx, http.MethodGet, "/health", nil)
	return err == nil
}

// CreateCollection creates a new vector collection.
func (c *Client) CreateCollection(ctx context.Context, cfg CollectionConfig) error {
	nShards := cfg.NShards
	if nShards <= 0 {
		nShards = 4
	}
	body := map[string]interface{}{
		"name":     cfg.Name,
		"dim":      cfg.Dim,
		"n_shards": nShards,
	}
	if cfg.NReplicas > 0 {
		body["n_replicas"] = cfg.NReplicas
	}
	if cfg.IndexType != "" {
		body["index_type"] = cfg.IndexType
	}
	if cfg.Metric != "" {
		body["metric"] = cfg.Metric
	}
	temporal := map[string]interface{}{}
	if cfg.HalfLifeMs > 0 {
		temporal["half_life_ms"] = cfg.HalfLifeMs
	}
	if cfg.TemporalBlend > 0 {
		temporal["temporal_blend"] = cfg.TemporalBlend
	}
	if cfg.StalenessThreshold > 0 {
		temporal["staleness_threshold"] = cfg.StalenessThreshold
	}
	if len(temporal) > 0 {
		body["temporal"] = temporal
	}
	if cfg.Supersession != nil {
		body["supersession"] = cfg.Supersession
	}
	if cfg.TextField != "" {
		body["text_field"] = cfg.TextField
	}
	if cfg.DisablePayloadIndex {
		body["payload_index"] = false
	}
	if cfg.CompactionRatio > 0 {
		body["compaction_ratio"] = cfg.CompactionRatio
	}
	_, err := c.do(ctx, http.MethodPost, "/v1/collections", body)
	return err
}

// DropCollection deletes a collection and all its vectors.
func (c *Client) DropCollection(ctx context.Context, name string) error {
	_, err := c.do(ctx, http.MethodDelete, "/v1/collections/"+url.PathEscape(name), nil)
	return err
}

// GetCollection returns stats for one collection.
func (c *Client) GetCollection(ctx context.Context, name string) (*CollectionInfo, error) {
	var info CollectionInfo
	if err := c.doData(ctx, http.MethodGet, "/v1/collections/"+url.PathEscape(name), nil, &info); err != nil {
		return nil, err
	}
	return &info, nil
}

// Count returns the number of live vectors in a collection.
func (c *Client) Count(ctx context.Context, collection string) (int, error) {
	info, err := c.GetCollection(ctx, collection)
	if err != nil {
		return 0, err
	}
	return info.NVectors, nil
}

// Upsert inserts or updates vectors in a collection.
// TimestampMs, when set, becomes the record's created_at.
func (c *Client) Upsert(ctx context.Context, collection string, vectors []Vector) error {
	mapped := make([]map[string]interface{}, len(vectors))
	for i, v := range vectors {
		m := map[string]interface{}{
			"id":        v.ID,
			"embedding": v.Embedding,
		}
		if len(v.Payload) > 0 {
			m["payload"] = v.Payload
		}
		if len(v.Edges) > 0 {
			m["edges"] = v.Edges
		}
		if v.TimestampMs != nil {
			m["created_at"] = *v.TimestampMs
		}
		mapped[i] = m
	}
	_, err := c.do(ctx, http.MethodPost, c.colPath(collection, "upsert"),
		map[string]interface{}{"vectors": mapped})
	return err
}

// Search queries a collection for the nearest vectors to the given embedding.
func (c *Client) Search(ctx context.Context, collection string, embedding []float32, opts SearchOptions) (*SearchResponse, error) {
	topK := opts.TopK
	if topK <= 0 {
		topK = 10
	}
	if embedding == nil && opts.Text == "" {
		return nil, errors.New("tidevec: Search needs an embedding or Text")
	}
	body := map[string]interface{}{
		"top_k":                      topK,
		"include_trace":              opts.IncludeTrace,
		"include_staleness_warnings": opts.StalenessWarnings,
	}
	if embedding != nil {
		body["vector"] = embedding
	}
	if opts.TemporalBlend != nil {
		body["temporal_blend"] = *opts.TemporalBlend
	}
	if opts.Mode != "" {
		body["mode"] = opts.Mode
	}
	if opts.CausalHops > 0 {
		body["causal_hops"] = opts.CausalHops
	}
	if err := setFilter(body, opts.Filter, opts.Where, opts.AsOf); err != nil {
		return nil, err
	}
	if opts.Text != "" {
		body["text"] = opts.Text
		if opts.Hybrid != nil {
			body["hybrid"] = opts.Hybrid
		}
	}
	if opts.EfSearch > 0 {
		body["ef_search"] = opts.EfSearch
	}
	if opts.SupersededPenalty > 0 {
		body["superseded_penalty"] = opts.SupersededPenalty
	}
	var res SearchResponse
	if err := c.doData(ctx, http.MethodPost, c.colPath(collection, "search"), body, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

// Delete removes vectors by id and/or by a "key=value" payload filter and
// returns how many were removed. Pass nil ids or "" filter to skip either.
func (c *Client) Delete(ctx context.Context, collection string, ids []string, filter string) (int, error) {
	if ids == nil && filter == "" {
		return 0, errors.New("tidevec: Delete needs ids or a filter")
	}
	body := map[string]interface{}{}
	if ids != nil {
		body["ids"] = ids
	}
	if filter != "" {
		if k, _, ok := strings.Cut(filter, "="); !ok || k == "" {
			return 0, fmt.Errorf("tidevec: filter must look like key=value, got %q", filter)
		}
		body["filter"] = filter
	}
	var out struct {
		Deleted int `json:"deleted"`
	}
	err := c.doData(ctx, http.MethodPost, c.colPath(collection, "delete"), body, &out)
	return out.Deleted, err
}

// DeleteVectors removes specific vectors from a collection by ID.
func (c *Client) DeleteVectors(ctx context.Context, collection string, ids []string) error {
	_, err := c.Delete(ctx, collection, ids, "")
	return err
}

// DeleteWhere removes every vector matching a rich filter and returns
// how many were removed.
func (c *Client) DeleteWhere(ctx context.Context, collection string, where Filter) (int, error) {
	if len(where) == 0 {
		return 0, errors.New("tidevec: DeleteWhere needs a non-empty filter")
	}
	var out struct {
		Deleted int `json:"deleted"`
	}
	err := c.doData(ctx, http.MethodPost, c.colPath(collection, "delete"),
		map[string]interface{}{"filter": where}, &out)
	return out.Deleted, err
}

// Get fetches one stored vector by id; it returns (nil, nil) if absent.
func (c *Client) Get(ctx context.Context, collection, id string, withVector bool) (*Record, error) {
	recs, _, err := c.GetMany(ctx, collection, []string{id}, withVector)
	if err != nil || len(recs) == 0 {
		return nil, err
	}
	return &recs[0], nil
}

// GetMany fetches stored vectors by id, in request order, and reports
// the ids that do not exist.
func (c *Client) GetMany(ctx context.Context, collection string, ids []string, withVector bool) ([]Record, []string, error) {
	var out struct {
		Vectors []Record `json:"vectors"`
		Missing []string `json:"missing"`
	}
	err := c.doData(ctx, http.MethodPost, c.colPath(collection, "get"),
		map[string]interface{}{"ids": ids, "with_vector": withVector}, &out)
	return out.Vectors, out.Missing, err
}

// Scroll returns one page of records in id order.
func (c *Client) Scroll(ctx context.Context, collection string, opts ScrollOptions) (*ScrollPage, error) {
	body := map[string]interface{}{
		"with_vector":  opts.WithVector,
		"with_payload": !opts.WithoutPayload,
	}
	if opts.Limit > 0 {
		body["limit"] = opts.Limit
	}
	if opts.Cursor != "" {
		body["cursor"] = opts.Cursor
	}
	if err := setFilter(body, opts.Filter, opts.Where, opts.AsOf); err != nil {
		return nil, err
	}
	var page ScrollPage
	if err := c.doData(ctx, http.MethodPost, c.colPath(collection, "scroll"), body, &page); err != nil {
		return nil, err
	}
	return &page, nil
}

// ListIDs returns the ids of every (matching) record, in order, starting
// after opts.Cursor. opts.Limit is the page size.
func (c *Client) ListIDs(ctx context.Context, collection string, opts ScrollOptions) ([]string, error) {
	if opts.Limit <= 0 {
		opts.Limit = 1000
	}
	opts.WithVector, opts.WithoutPayload = false, true
	var ids []string
	for {
		page, err := c.Scroll(ctx, collection, opts)
		if err != nil {
			return nil, err
		}
		for _, r := range page.Records {
			ids = append(ids, r.ID)
		}
		if page.NextCursor == "" {
			return ids, nil
		}
		opts.Cursor = page.NextCursor
	}
}

// CountWhere counts vectors matching a rich filter (nil = all) created
// at or before asOf (zero = now).
func (c *Client) CountWhere(ctx context.Context, collection string, where Filter, asOf time.Time) (int, error) {
	body := map[string]interface{}{}
	if err := setFilter(body, "", where, asOf); err != nil {
		return 0, err
	}
	var out struct {
		Count int `json:"count"`
	}
	err := c.doData(ctx, http.MethodPost, c.colPath(collection, "count"), body, &out)
	return out.Count, err
}

// SetAlias points alias at collection, atomically repointing it if it
// already exists. Every collection method accepts the alias as a name.
// It returns the collection the alias pointed at before ("" if new).
func (c *Client) SetAlias(ctx context.Context, alias, collection string) (string, error) {
	var out struct {
		Previous *string `json:"previous"`
	}
	err := c.doData(ctx, http.MethodPut, "/v1/aliases/"+url.PathEscape(alias),
		map[string]interface{}{"collection": collection}, &out)
	if out.Previous == nil {
		return "", err
	}
	return *out.Previous, err
}

// DeleteAlias removes an alias (the collection is untouched).
func (c *Client) DeleteAlias(ctx context.Context, alias string) error {
	_, err := c.do(ctx, http.MethodDelete, "/v1/aliases/"+url.PathEscape(alias), nil)
	return err
}

// ListAliases returns alias -> collection.
func (c *Client) ListAliases(ctx context.Context) (map[string]string, error) {
	var out struct {
		Aliases []struct {
			Alias      string `json:"alias"`
			Collection string `json:"collection"`
		} `json:"aliases"`
	}
	if err := c.doData(ctx, http.MethodGet, "/v1/aliases", nil, &out); err != nil {
		return nil, err
	}
	m := make(map[string]string, len(out.Aliases))
	for _, a := range out.Aliases {
		m[a.Alias] = a.Collection
	}
	return m, nil
}

// Compact rebuilds the collection's vector, payload and text indexes
// without the tombstones left by deletes and re-embeds. The rebuild runs
// in the background while reads and writes continue; with wait it returns
// once the rebuilt indexes are in place.
func (c *Client) Compact(ctx context.Context, collection string, wait bool) (*CompactionStatus, error) {
	var out CompactionStatus
	err := c.doData(ctx, http.MethodPost, c.colPath(collection, "compact"),
		map[string]interface{}{"wait": wait}, &out)
	if err != nil {
		return nil, err
	}
	return &out, nil
}

// CompactionStatus reports tombstones, index memory and compaction progress.
func (c *Client) CompactionStatus(ctx context.Context, collection string) (*CompactionStatus, error) {
	var out CompactionStatus
	if err := c.doData(ctx, http.MethodGet, c.colPath(collection, "compact"), nil, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

// AddEdges adds typed edges between existing vectors.
func (c *Client) AddEdges(ctx context.Context, collection string, edges []Edge) (int, error) {
	for i := range edges {
		if edges[i].Weight == 0 {
			edges[i].Weight = 1
		}
	}
	var out struct {
		Added int `json:"added"`
	}
	err := c.doData(ctx, http.MethodPost, c.colPath(collection, "edges"),
		map[string]interface{}{"edges": edges}, &out)
	return out.Added, err
}

// SetTemporal updates a collection's decay half-life and default blend.
func (c *Client) SetTemporal(ctx context.Context, collection string, halfLifeMs int64, blend float32) error {
	_, err := c.do(ctx, http.MethodPut, c.colPath(collection, "temporal"),
		map[string]interface{}{"half_life_ms": halfLifeMs, "temporal_blend": blend})
	return err
}

// Info returns server metadata from GET /v1/info.
func (c *Client) Info(ctx context.Context) (map[string]interface{}, error) {
	raw, err := c.do(ctx, http.MethodGet, "/v1/info", nil)
	if err != nil {
		return nil, err
	}
	var data map[string]interface{}
	return data, json.Unmarshal(raw, &data)
}

// ListCollections returns all collections on the server.
func (c *Client) ListCollections(ctx context.Context) ([]CollectionInfo, error) {
	var list []CollectionInfo
	err := c.doData(ctx, http.MethodGet, "/v1/collections", nil, &list)
	return list, err
}

// ClusterStatus returns this node's Raft state (experimental cluster mode).
func (c *Client) ClusterStatus(ctx context.Context) (*ClusterStatus, error) {
	var s ClusterStatus
	if err := c.doData(ctx, http.MethodGet, "/v1/cluster/status", nil, &s); err != nil {
		return nil, err
	}
	return &s, nil
}

// StartDrift begins a zero-downtime embedding model migration.
func (c *Client) StartDrift(ctx context.Context, collection, reembedURL string) error {
	_, err := c.do(ctx, http.MethodPost, c.colPath(collection, "drift/start"),
		map[string]interface{}{"reembed_url": reembedURL})
	return err
}

// DriftStatus polls migration progress.
func (c *Client) DriftStatus(ctx context.Context, collection string) (*DriftStatus, error) {
	var s DriftStatus
	if err := c.doData(ctx, http.MethodGet, c.colPath(collection, "drift/status"), nil, &s); err != nil {
		return nil, err
	}
	return &s, nil
}

// AbortDrift cancels an in-progress migration.
func (c *Client) AbortDrift(ctx context.Context, collection string) error {
	_, err := c.do(ctx, http.MethodPost, c.colPath(collection, "drift/abort"), map[string]interface{}{})
	return err
}

// TriggerBackup creates a manual snapshot (requires auth).
func (c *Client) TriggerBackup(ctx context.Context) (string, error) {
	var data struct {
		Snapshot string `json:"snapshot"`
	}
	err := c.doData(ctx, http.MethodPost, "/v1/admin/backup", map[string]interface{}{}, &data)
	return data.Snapshot, err
}

// ListBackups returns available snapshot filenames.
func (c *Client) ListBackups(ctx context.Context) ([]string, error) {
	var data struct {
		Snapshots []string `json:"snapshots"`
	}
	err := c.doData(ctx, http.MethodGet, "/v1/admin/backups", nil, &data)
	return data.Snapshots, err
}

// BackupManifest is a point-in-time recovery record.
type BackupManifest struct {
	Snapshot  string `json:"snapshot"`
	CreatedAt int64  `json:"created_at"`
	DataDir   string `json:"data_dir"`
	S3URI     string `json:"s3_uri"`
	GCSURI    string `json:"gcs_uri"`
}

// ListBackupManifests returns PITR manifest history.
func (c *Client) ListBackupManifests(ctx context.Context) ([]BackupManifest, error) {
	var data struct {
		Manifests []BackupManifest `json:"manifests"`
	}
	err := c.doData(ctx, http.MethodGet, "/v1/admin/backups/manifests", nil, &data)
	return data.Manifests, err
}

// RestoreBackup restores from a snapshot. Server restart required.
func (c *Client) RestoreBackup(ctx context.Context, snapshot string) (string, error) {
	var data struct {
		Message string `json:"message"`
	}
	err := c.doData(ctx, http.MethodPost, "/v1/admin/restore",
		map[string]interface{}{"snapshot": snapshot, "confirm": true}, &data)
	return data.Message, err
}

// --- internal helpers ---

func (c *Client) colPath(collection, op string) string {
	return "/v1/collections/" + url.PathEscape(collection) + "/" + op
}

func setFilter(body map[string]interface{}, legacy string, where Filter, asOf time.Time) error {
	switch {
	case where != nil:
		body["filter"] = where
	case legacy != "":
		if k, _, ok := strings.Cut(legacy, "="); !ok || k == "" {
			return fmt.Errorf("tidevec: filter must look like key=value, got %q", legacy)
		}
		body["filter"] = legacy
	}
	if !asOf.IsZero() {
		body["as_of"] = asOf.UnixMilli()
	}
	return nil
}

// envelope is the outer wrapper TideVec responses use:
// {"status":"ok","data":{...}} or {"status":"error","error":"...","leader":...}
type envelope struct {
	Status string          `json:"status"`
	Data   json.RawMessage `json:"data"`
	Error  string          `json:"error"`
	Leader string          `json:"leader"`
}

// doData performs a request and decodes the envelope's data field into out.
func (c *Client) doData(ctx context.Context, method, path string, payload, out interface{}) error {
	raw, err := c.do(ctx, method, path, payload)
	if err != nil {
		return err
	}
	var env envelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return fmt.Errorf("tidevec: decode response: %w", err)
	}
	if len(env.Data) == 0 || out == nil {
		return nil
	}
	if err := json.Unmarshal(env.Data, out); err != nil {
		return fmt.Errorf("tidevec: decode data: %w", err)
	}
	return nil
}

// do sends a request, retrying on 429 and following cluster-mode leader
// redirects (421), and returns the raw response body.
func (c *Client) do(ctx context.Context, method, path string, payload interface{}) ([]byte, error) {
	var body []byte
	if payload != nil {
		b, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("tidevec: marshal error: %w", err)
		}
		body = b
	}

	base := c.baseURL
	rateRetries, hops := 0, 0
	redirectDeadline := time.Now().Add(10 * time.Second)
	for {
		data, status, err := c.send(ctx, method, base+path, body)
		if err != nil {
			if base != c.baseURL && time.Now().Before(redirectDeadline) {
				base = c.baseURL // stale leader hint pointing at a dead node
				continue
			}
			return nil, err
		}
		if status < 400 {
			return data, nil
		}
		apiErr := &APIError{Status: status, Message: string(data)}
		var env envelope
		if json.Unmarshal(data, &env) == nil && env.Error != "" {
			apiErr.Message, apiErr.Leader = env.Error, env.Leader
		}
		switch {
		case status == http.StatusTooManyRequests && rateRetries < c.maxRetries:
			rateRetries++
			if err := sleepCtx(ctx, time.Duration(min(rateRetries*rateRetries, 8))*time.Second); err != nil {
				return nil, err
			}
		case status == http.StatusMisdirectedRequest && time.Now().Before(redirectDeadline):
			hops++
			leader := strings.TrimRight(apiErr.Leader, "/")
			if leader != "" && leader != base && hops <= 2 {
				base = leader
				continue
			}
			if err := sleepCtx(ctx, time.Duration(min(50*hops, 500))*time.Millisecond); err != nil {
				return nil, err
			}
			if leader != "" {
				base = leader
			} else {
				base = c.baseURL
			}
		default:
			return nil, apiErr
		}
	}
}

func (c *Client) send(ctx context.Context, method, fullURL string, body []byte) ([]byte, int, error) {
	var rdr io.Reader
	if body != nil {
		rdr = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, fullURL, rdr)
	if err != nil {
		return nil, 0, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.apiKey != "" {
		req.Header.Set("X-Api-Key", c.apiKey)
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("tidevec: request failed: %w", err)
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, resp.StatusCode, fmt.Errorf("tidevec: read response: %w", err)
	}
	return data, resp.StatusCode, nil
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}
