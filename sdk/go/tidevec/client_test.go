package tidevec

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// Integration tests against a live tidevec-server. Set TIDEVEC_TEST_ADDR
// to use a running server, or TIDEVEC_SERVER_BIN (default:
// ../../../build/tidevec-server) to have the tests start one.

var testAddr string

func TestMain(m *testing.M) {
	os.Exit(run(m))
}

func run(m *testing.M) int {
	if addr := os.Getenv("TIDEVEC_TEST_ADDR"); addr != "" {
		testAddr = addr
		return m.Run()
	}
	bin := os.Getenv("TIDEVEC_SERVER_BIN")
	if bin == "" {
		bin, _ = filepath.Abs("../../../build/tidevec-server")
	}
	if _, err := os.Stat(bin); err != nil {
		fmt.Println("skipping integration tests: tidevec-server not found at", bin)
		return 0
	}
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		fmt.Println("cannot pick a port:", err)
		return 1
	}
	port := l.Addr().(*net.TCPAddr).Port
	l.Close()
	dir, _ := os.MkdirTemp("", "tidevec-go-test")
	defer os.RemoveAll(dir)

	cmd := exec.Command(bin, "--port", fmt.Sprint(port), "--data-dir", dir,
		"--no-require-auth", "--quiet", "--rate-limit", "100000")
	if err := cmd.Start(); err != nil {
		fmt.Println("cannot start server:", err)
		return 1
	}
	defer func() { _ = cmd.Process.Kill(); _ = cmd.Wait() }()

	testAddr = fmt.Sprintf("127.0.0.1:%d", port)
	c, _ := New(testAddr)
	deadline := time.Now().Add(10 * time.Second)
	for !c.Ping(context.Background()) {
		if time.Now().After(deadline) {
			fmt.Println("server did not start")
			return 1
		}
		time.Sleep(100 * time.Millisecond)
	}
	return m.Run()
}

func client(t *testing.T) *Client {
	t.Helper()
	c, err := New(testAddr, ClientConfig{TimeoutMs: 10_000})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func randVec(r *rand.Rand, dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = r.Float32()*2 - 1
	}
	return v
}

func newCollection(t *testing.T, c *Client, cfg CollectionConfig) {
	t.Helper()
	ctx := context.Background()
	_ = c.DropCollection(ctx, cfg.Name)
	if err := c.CreateCollection(ctx, cfg); err != nil {
		t.Fatalf("CreateCollection: %v", err)
	}
	t.Cleanup(func() { _ = c.DropCollection(context.Background(), cfg.Name) })
}

func TestSearchAndTemporalDecay(t *testing.T) {
	ctx := context.Background()
	c := client(t)
	r := rand.New(rand.NewSource(1))
	const col, dim = "go_search", 8
	newCollection(t, c, CollectionConfig{Name: col, Dim: dim, HalfLifeMs: HalfLifeOneWeek, TemporalBlend: 0.3})

	query := randVec(r, dim)
	old := time.Now().UnixMilli() - 30*24*3600*1000
	vecs := []Vector{
		{ID: "fresh", Embedding: query, Payload: map[string]string{"lang": "go"}},
		{ID: "stale", Embedding: query, TimestampMs: &old},
	}
	for i := 0; i < 10; i++ {
		vecs = append(vecs, Vector{ID: fmt.Sprint("n", i), Embedding: randVec(r, dim)})
	}
	if err := c.Upsert(ctx, col, vecs); err != nil {
		t.Fatal(err)
	}
	if n, err := c.Count(ctx, col); err != nil || n != 12 {
		t.Fatalf("Count = %d, %v; want 12", n, err)
	}

	res, err := c.Search(ctx, col, query, SearchOptions{TopK: 12, StalenessWarnings: true})
	if err != nil {
		t.Fatal(err)
	}
	if res.Hits[0].ID != "fresh" || res.Hits[0].Payload["lang"] != "go" {
		t.Fatalf("top hit = %+v, want fresh", res.Hits[0])
	}
	var stale *SearchHit
	for i := range res.Hits {
		if res.Hits[i].ID == "stale" {
			stale = &res.Hits[i]
		}
	}
	if stale == nil || !stale.StalenessWarning || stale.Score >= res.Hits[0].Score {
		t.Fatalf("stale hit should rank below fresh with a warning: %+v", stale)
	}

	// Blend(0) is semantic-only: identical embeddings tie on score.
	res, err = c.Search(ctx, col, query, SearchOptions{TopK: 2, TemporalBlend: Blend(0)})
	if err != nil {
		t.Fatal(err)
	}
	if res.Hits[0].Score != res.Hits[1].Score {
		t.Fatalf("semantic-only scores differ: %v vs %v", res.Hits[0].Score, res.Hits[1].Score)
	}

	res, err = c.Search(ctx, col, query, SearchOptions{TopK: 5, Filter: "lang=go"})
	if err != nil || len(res.Hits) != 1 || res.Hits[0].ID != "fresh" {
		t.Fatalf("filtered search = %+v, %v", res, err)
	}
}

func TestSupersession(t *testing.T) {
	ctx := context.Background()
	c := client(t)
	const col = "go_supersession"
	newCollection(t, c, CollectionConfig{
		Name: col, Dim: 4,
		Supersession: &SupersessionConfig{MatchField: "doc", SimilarityThreshold: 0.9},
	})
	v1 := time.Now().UnixMilli() - 3600_000
	if err := c.Upsert(ctx, col, []Vector{{
		ID: "policy_v1", Embedding: []float32{1, 0.1, 0, 0},
		Payload: map[string]string{"doc": "refund"}, TimestampMs: &v1,
	}}); err != nil {
		t.Fatal(err)
	}
	if err := c.Upsert(ctx, col, []Vector{{
		ID: "policy_v2", Embedding: []float32{1, 0.12, 0, 0},
		Payload: map[string]string{"doc": "refund"},
	}}); err != nil {
		t.Fatal(err)
	}

	info, err := c.GetCollection(ctx, col)
	if err != nil || info.Supersession == nil || !info.Supersession.Enabled ||
		info.Supersession.MatchField != "doc" {
		t.Fatalf("GetCollection supersession = %+v, %v", info, err)
	}

	res, err := c.Search(ctx, col, []float32{1, 0.1, 0, 0}, SearchOptions{TopK: 2, TemporalBlend: Blend(0)})
	if err != nil {
		t.Fatal(err)
	}
	byID := map[string]SearchHit{}
	for _, h := range res.Hits {
		byID[h.ID] = h
	}
	if !byID["policy_v1"].IsSuperseded() || byID["policy_v1"].SupersededBy[0] != "policy_v2" {
		t.Fatalf("policy_v1 should be superseded by policy_v2: %+v", byID["policy_v1"])
	}
	if byID["policy_v2"].IsSuperseded() {
		t.Fatalf("policy_v2 should be current: %+v", byID["policy_v2"])
	}

	// A penalty pushes the superseded version down.
	res, err = c.Search(ctx, col, []float32{1, 0.1, 0, 0},
		SearchOptions{TopK: 2, TemporalBlend: Blend(0), SupersededPenalty: 0.5})
	if err != nil || res.Hits[0].ID != "policy_v2" {
		t.Fatalf("with penalty top hit = %+v, %v", res, err)
	}
}

func TestDeleteByFilterAndEdges(t *testing.T) {
	ctx := context.Background()
	c := client(t)
	r := rand.New(rand.NewSource(2))
	const col = "go_delete"
	newCollection(t, c, CollectionConfig{Name: col, Dim: 4, NShards: 2})

	var vecs []Vector
	for i := 0; i < 10; i++ {
		grp := "keep"
		if i%2 == 0 {
			grp = "drop"
		}
		vecs = append(vecs, Vector{ID: fmt.Sprint("v", i), Embedding: randVec(r, 4),
			Payload: map[string]string{"grp": grp}})
	}
	if err := c.Upsert(ctx, col, vecs); err != nil {
		t.Fatal(err)
	}
	if n, err := c.Delete(ctx, col, nil, "grp=drop"); err != nil || n != 5 {
		t.Fatalf("Delete by filter = %d, %v; want 5", n, err)
	}
	if n, err := c.Delete(ctx, col, []string{"v1", "missing"}, ""); err != nil || n != 1 {
		t.Fatalf("Delete by ids = %d, %v; want 1", n, err)
	}
	if n, _ := c.Count(ctx, col); n != 4 {
		t.Fatalf("Count after deletes = %d, want 4", n)
	}
	if _, err := c.Delete(ctx, col, nil, "no-equals"); err == nil {
		t.Fatal("malformed filter should be rejected")
	}

	added, err := c.AddEdges(ctx, col, []Edge{{Src: "v3", Tgt: "v5", Type: EdgeContradicts}})
	if err != nil || added != 1 {
		t.Fatalf("AddEdges = %d, %v", added, err)
	}
	if err := c.SetTemporal(ctx, col, HalfLifeOneDay, 0.5); err != nil {
		t.Fatal(err)
	}
}

func TestErrorsAndStatus(t *testing.T) {
	ctx := context.Background()
	c := client(t)
	_, err := c.Search(ctx, "does_not_exist", []float32{1}, SearchOptions{})
	if !IsNotFound(err) {
		t.Fatalf("want 404 APIError, got %v", err)
	}
	var ae *APIError
	if !errors.As(err, &ae) || ae.Message == "" {
		t.Fatalf("want *APIError with message, got %#v", err)
	}
	if err := c.DropCollection(ctx, "does_not_exist"); !IsNotFound(err) {
		t.Fatalf("drop missing: want 404, got %v", err)
	}
	s, err := c.ClusterStatus(ctx)
	if err != nil || s.Mode != "standalone" {
		t.Fatalf("ClusterStatus = %+v, %v", s, err)
	}
	info, err := c.Info(ctx)
	if err != nil || info["version"] != "0.3.0" {
		t.Fatalf("Info = %v, %v", info, err)
	}
}
