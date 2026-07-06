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
	"os"
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
	SemanticScore    float32           `json:"vector_score"`
	TemporalScore    float32           `json:"temporal_score"`
	CreatedAt        int64             `json:"created_at"`
	Payload          map[string]string `json:"payload"`
	StalenessWarning bool              `json:"staleness_warning"`
	StalenessReason  string            `json:"staleness_reason"`
}

// SearchResponse wraps search results with an optional trace.
type SearchResponse struct {
	Hits  []SearchHit `json:"results"` // server field is "results" not "hits"
	Count int         `json:"count"`
}

// apiEnvelope is the outer wrapper all TideVec responses use:
// {"status":"ok","data":{...}} or {"status":"error","error":"..."}
type apiEnvelope struct {
	Status string          `json:"status"`
	Data   json.RawMessage `json:"data"`
	Error  string          `json:"error"`
}

// ClientConfig holds optional settings for the HTTP client.
type ClientConfig struct {
	APIKey     string
	TLS        bool   // use https:// instead of http://
	TimeoutMs  int
	MaxRetries int
}

// DriftStatus reports zero-downtime model migration progress.
type DriftStatus struct {
	Collection    string  `json:"collection"`
	Phase         string  `json:"phase"`
	TotalVectors  int     `json:"total_vectors"`
	Migrated      int     `json:"migrated"`
	Skipped       int     `json:"skipped"`
	PctComplete   float64 `json:"pct_complete"`
	Error         string  `json:"error"`
}

// CollectionInfo describes a collection on the server.
type CollectionInfo struct {
	Name      string `json:"name"`
	NVectors  int    `json:"n_vectors"`
	NShards   int    `json:"n_shards"`
	Dim       int    `json:"dim"`
	IndexType string `json:"index_type"`
	Metric    string `json:"metric"`
	Backend   string `json:"backend"`
}

// Client is a TideVec REST client.
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

	c := &Client{
		baseURL:    scheme + "://" + hostPort,
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
	// Default n_shards to 4 if not set
	nShards := cfg.NShards
	if nShards <= 0 {
		nShards = 4
	}
	// Default staleness_threshold to 0.20 if not set
	stalenessThreshold := cfg.StalenessThreshold
	if stalenessThreshold <= 0 {
		stalenessThreshold = 0.20
	}
	// Server expects temporal config nested under "temporal" key
	body := map[string]interface{}{
		"name":       cfg.Name,
		"dim":        cfg.Dim,
		"n_shards":   nShards,
		"n_replicas": cfg.NReplicas,
		"index_type": cfg.IndexType,
		"temporal": map[string]interface{}{
			"half_life_ms":        cfg.HalfLifeMs,
			"temporal_blend":      cfg.TemporalBlend,
			"staleness_threshold": stalenessThreshold,
		},
	}
	_, err := c.postJSON(ctx, "/v1/collections", body)
	return err
}

// DropCollection deletes a collection and all its vectors.
func (c *Client) DropCollection(ctx context.Context, name string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete,
		c.baseURL+"/v1/collections/"+name, nil)
	if err != nil {
		return err
	}
	if c.apiKey != "" {
		req.Header.Set("X-Api-Key", c.apiKey)
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("tidevec: drop collection failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("tidevec: drop collection error %d: %s", resp.StatusCode, string(body))
	}
	return nil
}

// Upsert inserts or updates vectors in a collection.
// If a vector has TimestampMs set, it is remapped to "created_at"
// which is the field name the server reads for temporal decay.
func (c *Client) Upsert(ctx context.Context, collection string, vectors []Vector) error {
	// Build a list of plain maps so we can remap timestamp_ms → created_at.
	// The server reads "created_at", not "timestamp_ms".
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
			m["created_at"] = *v.TimestampMs // remap to server field name
		}
		mapped[i] = m
	}
	body := map[string]interface{}{"vectors": mapped}
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
	raw, err := c.postJSON(ctx, "/v1/collections/"+collection+"/search", body)
	if err != nil {
		return nil, err
	}

	// Unwrap the {"status":"ok","data":{...}} envelope
	var env apiEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, fmt.Errorf("tidevec: failed to decode envelope: %w", err)
	}
	if env.Status != "ok" {
		return nil, fmt.Errorf("tidevec: server error: %s", env.Error)
	}

	// Parse the inner data object — server returns "results" not "hits"
	var result SearchResponse
	if err := json.Unmarshal(env.Data, &result); err != nil {
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

// Info returns server metadata from GET /v1/info.
func (c *Client) Info(ctx context.Context) (map[string]interface{}, error) {
	raw, err := c.getJSON(ctx, "/v1/info")
	if err != nil {
		return nil, err
	}
	var env apiEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, err
	}
	var data map[string]interface{}
	if err := json.Unmarshal(env.Data, &data); err != nil {
		return nil, err
	}
	return data, nil
}

// ListCollections returns all collections on the server.
func (c *Client) ListCollections(ctx context.Context) ([]CollectionInfo, error) {
	raw, err := c.getJSON(ctx, "/v1/collections")
	if err != nil {
		return nil, err
	}
	var env apiEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, err
	}
	var list []CollectionInfo
	if err := json.Unmarshal(env.Data, &list); err != nil {
		return nil, err
	}
	return list, nil
}

// StartDrift begins a zero-downtime embedding model migration.
func (c *Client) StartDrift(ctx context.Context, collection, reembedURL string) error {
	body := map[string]interface{}{
		"reembed_url": reembedURL,
	}
	_, err := c.postJSON(ctx, "/v1/collections/"+collection+"/drift/start", body)
	return err
}

// DriftStatus polls migration progress.
func (c *Client) DriftStatus(ctx context.Context, collection string) (*DriftStatus, error) {
	raw, err := c.getJSON(ctx, "/v1/collections/"+collection+"/drift/status")
	if err != nil {
		return nil, err
	}
	var env apiEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, err
	}
	var status DriftStatus
	if err := json.Unmarshal(env.Data, &status); err != nil {
		return nil, err
	}
	return &status, nil
}

// AbortDrift cancels an in-progress migration.
func (c *Client) AbortDrift(ctx context.Context, collection string) error {
	_, err := c.postJSON(ctx, "/v1/collections/"+collection+"/drift/abort", map[string]interface{}{})
	return err
}

// TriggerBackup creates a manual snapshot (requires auth).
func (c *Client) TriggerBackup(ctx context.Context) (string, error) {
	raw, err := c.postJSON(ctx, "/v1/admin/backup", map[string]interface{}{})
	if err != nil {
		return "", err
	}
	var env apiEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return "", err
	}
	var data struct {
		Snapshot string `json:"snapshot"`
	}
	if err := json.Unmarshal(env.Data, &data); err != nil {
		return "", err
	}
	return data.Snapshot, nil
}

// ListBackups returns available snapshot filenames.
func (c *Client) ListBackups(ctx context.Context) ([]string, error) {
	raw, err := c.getJSON(ctx, "/v1/admin/backups")
	if err != nil {
		return nil, err
	}
	var env apiEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, err
	}
	var data struct {
		Snapshots []string `json:"snapshots"`
	}
	if err := json.Unmarshal(env.Data, &data); err != nil {
		return nil, err
	}
	return data.Snapshots, nil
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
	raw, err := c.getJSON(ctx, "/v1/admin/backups/manifests")
	if err != nil {
		return nil, err
	}
	var env apiEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return nil, err
	}
	var data struct {
		Manifests []BackupManifest `json:"manifests"`
	}
	if err := json.Unmarshal(env.Data, &data); err != nil {
		return nil, err
	}
	return data.Manifests, nil
}

// RestoreBackup restores from a snapshot. Server restart required.
func (c *Client) RestoreBackup(ctx context.Context, snapshot string) (string, error) {
	raw, err := c.postJSON(ctx, "/v1/admin/restore", map[string]interface{}{
		"snapshot": snapshot,
		"confirm":  true,
	})
	if err != nil {
		return "", err
	}
	var env apiEnvelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return "", err
	}
	var data struct {
		Message string `json:"message"`
	}
	if err := json.Unmarshal(env.Data, &data); err != nil {
		return "", err
	}
	return data.Message, nil
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

func (c *Client) getJSON(ctx context.Context, path string) ([]byte, error) {
	resp, err := c.get(ctx, path)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("tidevec: server error %d: %s", resp.StatusCode, string(data))
	}
	return data, nil
}

func (c *Client) postJSON(ctx context.Context, path string, payload interface{}) ([]byte, error) {
	b, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("tidevec: marshal error: %w", err)
	}

	var lastErr error
	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			time.Sleep(time.Duration(min(attempt*attempt, 8)) * time.Second)
		}
		data, status, err := c.doPost(ctx, path, b)
		if err != nil {
			return nil, err
		}
		if status == http.StatusTooManyRequests {
			lastErr = fmt.Errorf("tidevec: rate limit exceeded (429)")
			continue
		}
		if status >= 400 {
			return nil, fmt.Errorf("tidevec: server error %d: %s", status, string(data))
		}
		return data, nil
	}
	return nil, lastErr
}

func (c *Client) doPost(ctx context.Context, path string, body []byte) ([]byte, int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
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
