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
//	results, err := db.Search(ctx, "docs", queryVec, tidevec.SearchOptions{TopK: 10})
package tidevec

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
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
	EdgeCauses      EdgeType = "CAUSES"
	EdgeContradicts EdgeType = "CONTRADICTS"
	EdgeUpdates     EdgeType = "UPDATES"
	EdgeRelatedTo   EdgeType = "RELATED_TO"
	EdgeEntityOf    EdgeType = "ENTITY_OF"
	EdgeSupports    EdgeType = "SUPPORTS"
)

// CausalEdge links two vectors with a typed relationship.
type CausalEdge struct {
	TargetID string   `json:"target_id"`
	Type     EdgeType `json:"type"`
	Weight   float32  `json:"weight"`
}

// Vector is a record stored in a TideVec collection.
type Vector struct {
	ID          string            `json:"id"`
	Embedding   []float32         `json:"embedding"`
	Payload     map[string]string `json:"payload,omitempty"`
	Edges       []CausalEdge      `json:"edges,omitempty"`
	TimestampMs *int64            `json:"timestamp_ms,omitempty"`
}

// CollectionConfig holds settings for a new collection.
type CollectionConfig struct {
	Name               string  `json:"name"`
	Dim                int     `json:"dim"`
	HalfLifeMs         int64   `json:"half_life_ms,omitempty"`
	TemporalBlend      float32 `json:"temporal_blend,omitempty"`
	NShards            int     `json:"n_shards,omitempty"`
	NReplicas          int     `json:"n_replicas,omitempty"`
	IndexType          string  `json:"index_type,omitempty"`
	StalenessThreshold float32 `json:"staleness_threshold,omitempty"`
}

// SearchOptions controls how a search query is executed.
type SearchOptions struct {
	TopK              int     `json:"top_k"`
	TemporalBlend     float32 `json:"temporal_blend,omitempty"`
	Mode              string  `json:"mode,omitempty"`
	CausalHops        int     `json:"causal_hops,omitempty"`
	IncludeTrace      bool    `json:"include_trace,omitempty"`
	StalenessWarnings bool    `json:"include_staleness_warnings,omitempty"`
}

// SearchHit is a single result from a search query.
type SearchHit struct {
	ID               string            `json:"id"`
	Score            float32           `json:"score"`
	SemanticScore    float32           `json:"semantic_score"`
	TemporalScore    float32           `json:"temporal_score"`
	Payload          map[string]string `json:"payload"`
	StalenessWarning bool              `json:"staleness_warning"`
	StalenessReason  string            `json:"staleness_reason"`
}

// SearchResponse wraps search results with an optional trace.
type SearchResponse struct {
	Hits  []SearchHit `json:"hits"`
	Trace *struct {
		Algorithm      string  `json:"algorithm"`
		LatencyMs      float64 `json:"latency_ms"`
		RecallEstimate float32 `json:"recall_estimate"`
		DeviceUsed     string  `json:"device_used"`
	} `json:"trace,omitempty"`
}

// ClientConfig holds optional settings for the HTTP client.
type ClientConfig struct {
	APIKey     string
	TimeoutMs  int
	MaxRetries int
}

// Client is a TideVec REST client.
type Client struct {
	baseURL    string
	httpClient *http.Client
	apiKey     string
	maxRetries int
}

// New creates a TideVec client connected to the given host:port.
func New(hostPort string, cfg ...ClientConfig) (*Client, error) {
	timeout := 30 * time.Second
	apiKey := ""
	maxRetries := 3

	if len(cfg) > 0 {
		if cfg[0].TimeoutMs > 0 {
			timeout = time.Duration(cfg[0].TimeoutMs) * time.Millisecond
		}
		apiKey = cfg[0].APIKey
		if cfg[0].MaxRetries > 0 {
			maxRetries = cfg[0].MaxRetries
		}
	}

	c := &Client{
		baseURL:    "http://" + hostPort,
		httpClient: &http.Client{Timeout: timeout},
		apiKey:     apiKey,
		maxRetries: maxRetries,
	}
	return c, nil
}

// Close is a no-op for the HTTP client (provided for interface symmetry).
func (c *Client) Close() {}

// Ping checks whether the server is reachable.
func (c *Client) Ping(ctx context.Context) bool {
	resp, err := c.get(ctx, "/health")
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// CreateCollection creates a new vector collection.
func (c *Client) CreateCollection(ctx context.Context, cfg CollectionConfig) error {
	_, err := c.postJSON(ctx, "/v1/collections", cfg)
	return err
}

// DropCollection deletes a collection and all its vectors.
func (c *Client) DropCollection(ctx context.Context, name string) error {
	_, err := c.postJSON(ctx, "/v1/collections/"+name+"/delete", struct{}{})
	return err
}

// Upsert inserts or updates vectors in a collection.
func (c *Client) Upsert(ctx context.Context, collection string, vectors []Vector) error {
	body := map[string]interface{}{"vectors": vectors}
	_, err := c.postJSON(ctx, "/v1/collections/"+collection+"/upsert", body)
	return err
}

// Search queries a collection for the nearest vectors to the given embedding.
func (c *Client) Search(ctx context.Context, collection string, embedding []float32, opts SearchOptions) (*SearchResponse, error) {
	body := map[string]interface{}{
		"vector":                     embedding,
		"top_k":                      opts.TopK,
		"temporal_blend":             opts.TemporalBlend,
		"mode":                       opts.Mode,
		"causal_hops":                opts.CausalHops,
		"include_trace":              opts.IncludeTrace,
		"include_staleness_warnings": opts.StalenessWarnings,
	}
	data, err := c.postJSON(ctx, "/v1/collections/"+collection+"/search", body)
	if err != nil {
		return nil, err
	}
	var result SearchResponse
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, fmt.Errorf("tidevec: failed to decode search response: %w", err)
	}
	return &result, nil
}

// DeleteVectors removes specific vectors from a collection by ID.
func (c *Client) DeleteVectors(ctx context.Context, collection string, ids []string) error {
	body := map[string]interface{}{"ids": ids}
	_, err := c.postJSON(ctx, "/v1/collections/"+collection+"/delete", body)
	return err
}

// --- internal helpers ---

func (c *Client) get(ctx context.Context, path string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+path, nil)
	if err != nil {
		return nil, err
	}
	if c.apiKey != "" {
		req.Header.Set("X-Api-Key", c.apiKey)
	}
	return c.httpClient.Do(req)
}

func (c *Client) postJSON(ctx context.Context, path string, payload interface{}) ([]byte, error) {
	b, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("tidevec: marshal error: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		req.Header.Set("X-Api-Key", c.apiKey)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("tidevec: request failed: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("tidevec: read response: %w", err)
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("tidevec: server error %d: %s", resp.StatusCode, string(data))
	}
	return data, nil
}
