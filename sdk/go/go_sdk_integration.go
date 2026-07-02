// go_sdk_test.go — TideVec Go SDK integration test
// Run with: go run go_sdk_test.go
// Requires: TideVec server running at localhost:6399
package main

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"time"

	tidevec "github.com/ashishodu2023/TideVec/sdk/go/tidevec"
)

func randVec(dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rand.Float32()*2 - 1
	}
	return v
}

func check(err error, msg string) {
	if err != nil {
		fmt.Fprintf(os.Stderr, "FAIL: %s: %v\n", msg, err)
		os.Exit(1)
	}
}

func assert(cond bool, msg string) {
	if !cond {
		fmt.Fprintf(os.Stderr, "FAIL: %s\n", msg)
		os.Exit(1)
	}
	fmt.Printf("  PASS: %s\n", msg)
}

func main() {
	ctx := context.Background()
	const col = "go_sdk_test"
	const dim = 8

	fmt.Println("=== TideVec Go SDK Integration Test ===")
	fmt.Println("Connecting to localhost:6399...")

	db, err := tidevec.New("localhost:6399")
	check(err, "New client")

	// 1. Ping
	assert(db.Ping(ctx), "Ping returns true — server reachable")

	// 2. Create collection (drop first if exists)
	_ = db.DropCollection(ctx, col) // ignore error if not exists
	err = db.CreateCollection(ctx, tidevec.CollectionConfig{
		Name:          col,
		Dim:           dim,
		HalfLifeMs:    tidevec.HalfLifeOneWeek,
		TemporalBlend: 0.3,
	})
	check(err, "CreateCollection")
	fmt.Println("  PASS: CreateCollection")

	// 3. Upsert vectors
	query := randVec(dim)
	vectors := []tidevec.Vector{
		{ID: "go_v1", Embedding: query,       Payload: map[string]string{"lang": "go", "rank": "1"}},
		{ID: "go_v2", Embedding: randVec(dim), Payload: map[string]string{"lang": "go", "rank": "2"}},
		{ID: "go_v3", Embedding: randVec(dim), Payload: map[string]string{"lang": "go", "rank": "3"}},
		{ID: "go_v4", Embedding: randVec(dim), Payload: map[string]string{"lang": "go", "rank": "4"}},
		{ID: "go_v5", Embedding: randVec(dim), Payload: map[string]string{"lang": "go", "rank": "5"}},
	}

	err = db.Upsert(ctx, col, vectors)
	check(err, "Upsert 5 vectors")
	fmt.Println("  PASS: Upsert 5 vectors")

	// Small delay to let vectors be indexed
	time.Sleep(100 * time.Millisecond)

	// 4. Search — query is identical to go_v1's embedding, so it should rank first
	results, err := db.Search(ctx, col, query, tidevec.SearchOptions{
		TopK:              5,
		TemporalBlend:     0.3,
		StalenessWarnings: true,
	})
	check(err, "Search")
	assert(len(results.Hits) > 0, fmt.Sprintf("Search returned %d hits (want > 0)", len(results.Hits)))
	assert(results.Hits[0].ID == "go_v1", fmt.Sprintf("Top hit is go_v1 (got %s)", results.Hits[0].ID))
	assert(results.Hits[0].Score > 0.8, fmt.Sprintf("Top score > 0.8 (got %.4f)", results.Hits[0].Score))
	assert(results.Hits[0].TemporalScore > 0.99, fmt.Sprintf("Fresh vector temporal > 0.99 (got %.4f)", results.Hits[0].TemporalScore))
	assert(!results.Hits[0].StalenessWarning, "Top hit not stale (just inserted)")
	assert(results.Hits[0].Payload["lang"] == "go", "Payload.lang = go")
	fmt.Printf("  PASS: Search top hit = %s score=%.4f temporal=%.4f\n",
		results.Hits[0].ID, results.Hits[0].Score, results.Hits[0].TemporalScore)

	// 5. Temporal decay — insert a backdated vector and verify it scores lower
	thirtyDaysAgo := time.Now().UnixMilli() - (30 * 24 * 60 * 60 * 1000)
	backdatedVec := []tidevec.Vector{{
		ID:          "go_stale",
		Embedding:   query, // identical to query — purely tests temporal ranking
		Payload:     map[string]string{"note": "backdated 30 days"},
		TimestampMs: &thirtyDaysAgo,
	}}
	err = db.Upsert(ctx, col, backdatedVec)
	check(err, "Upsert backdated vector")
	fmt.Println("  PASS: Upsert backdated vector (30 days old)")

	results2, err := db.Search(ctx, col, query, tidevec.SearchOptions{
		TopK:              6,
		TemporalBlend:     0.3,
		StalenessWarnings: true,
	})
	check(err, "Search with backdated vector")

	// Find the stale hit
	var staleHit *tidevec.SearchHit
	var freshHit *tidevec.SearchHit
	for i := range results2.Hits {
		if results2.Hits[i].ID == "go_stale" {
			staleHit = &results2.Hits[i]
		}
		if results2.Hits[i].ID == "go_v1" {
			freshHit = &results2.Hits[i]
		}
	}
	assert(staleHit != nil, "Backdated vector found in results")
	assert(freshHit != nil, "Fresh vector found in results")
	assert(staleHit.StalenessWarning, "Backdated vector has staleness_warning=true")
	assert(staleHit.TemporalScore < 0.1, fmt.Sprintf("Backdated temporal < 0.1 (got %.4f)", staleHit.TemporalScore))
	assert(freshHit.Score > staleHit.Score, fmt.Sprintf(
		"Fresh (%.4f) outranks stale (%.4f) despite identical embedding",
		freshHit.Score, staleHit.Score))

	fmt.Printf("  PASS: Temporal decay — fresh=%.4f stale=%.4f (%.0f%% differential)\n",
		freshHit.Score, staleHit.Score,
		(freshHit.Score-staleHit.Score)/freshHit.Score*100)

	// 6. Delete vectors
	err = db.DeleteVectors(ctx, col, []string{"go_v1", "go_stale"})
	check(err, "DeleteVectors")
	fmt.Println("  PASS: DeleteVectors")

	// 7. Drop collection
	err = db.DropCollection(ctx, col)
	check(err, "DropCollection")
	fmt.Println("  PASS: DropCollection")

	fmt.Println("\n=== ALL TESTS PASSED ✓ ===")
	fmt.Println("Go SDK is fully functional against a live TideVec server")
}
