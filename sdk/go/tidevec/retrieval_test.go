package tidevec

import (
	"context"
	"fmt"
	"sort"
	"testing"
	"time"
)

const t0 int64 = 1_700_000_000_000

func seedRetrieval(t *testing.T, c *Client, col string) {
	t.Helper()
	newCollection(t, c, CollectionConfig{Name: col, Dim: 4, NShards: 2, TextField: "text"})
	var vecs []Vector
	for i := 0; i < 50; i++ {
		ts := t0 + int64(i)*1000
		lang := "de"
		if i%2 == 1 {
			lang = "en"
		}
		text := fmt.Sprintf("plain note %d", i)
		if i == 17 {
			text = "error code TV-4711 on shutdown"
		}
		vecs = append(vecs, Vector{
			ID:          fmt.Sprintf("r%02d", i),
			Embedding:   []float32{1, float32(i) / 50, 0, 0},
			TimestampMs: &ts,
			Payload:     map[string]string{"lang": lang, "year": fmt.Sprint(2000 + i), "text": text},
		})
	}
	if err := c.Upsert(context.Background(), col, vecs); err != nil {
		t.Fatal(err)
	}
}

func hitIDs(res *SearchResponse) []string {
	var ids []string
	for _, h := range res.Hits {
		ids = append(ids, h.ID)
	}
	sort.Strings(ids)
	return ids
}

func TestGetScrollAndCount(t *testing.T) {
	ctx := context.Background()
	c := client(t)
	const col = "go_retrieval"
	seedRetrieval(t, c, col)

	info, err := c.GetCollection(ctx, col)
	if err != nil || info.TextField != "text" || !info.PayloadIndex {
		t.Fatalf("info = %+v, %v", info, err)
	}

	rec, err := c.Get(ctx, col, "r07", true)
	if err != nil || rec == nil || rec.Payload["lang"] != "en" || len(rec.Embedding) != 4 ||
		rec.CreatedAt != t0+7000 {
		t.Fatalf("Get = %+v, %v", rec, err)
	}
	if rec, err := c.Get(ctx, col, "nope", false); err != nil || rec != nil {
		t.Fatalf("Get(missing) = %+v, %v", rec, err)
	}
	recs, missing, err := c.GetMany(ctx, col, []string{"r03", "zz", "r01"}, false)
	if err != nil || len(recs) != 2 || recs[0].ID != "r03" || recs[1].ID != "r01" ||
		len(missing) != 1 || missing[0] != "zz" || recs[0].Embedding != nil {
		t.Fatalf("GetMany = %+v %v %v", recs, missing, err)
	}

	var seen []string
	opts := ScrollOptions{Limit: 16}
	pages := 0
	for {
		page, err := c.Scroll(ctx, col, opts)
		if err != nil {
			t.Fatal(err)
		}
		pages++
		for _, r := range page.Records {
			seen = append(seen, r.ID)
		}
		if page.NextCursor == "" {
			break
		}
		opts.Cursor = page.NextCursor
	}
	if pages != 4 || len(seen) != 50 || !sort.StringsAreSorted(seen) {
		t.Fatalf("scroll: %d pages, %d ids", pages, len(seen))
	}

	ids, err := c.ListIDs(ctx, col, ScrollOptions{Limit: 7, Where: Filter{"lang": "en"}})
	if err != nil || len(ids) != 25 || ids[0] != "r01" {
		t.Fatalf("ListIDs = %v, %v", ids, err)
	}

	n, err := c.CountWhere(ctx, col, Filter{"$or": []Filter{
		{"lang": "de"}, {"year": Filter{"$gte": 2045}}}}, time.Time{})
	if err != nil || n != 28 {
		t.Fatalf("CountWhere = %d, %v; want 28", n, err)
	}
	n, err = c.CountWhere(ctx, col, nil, time.UnixMilli(t0+9000))
	if err != nil || n != 10 {
		t.Fatalf("CountWhere(as_of) = %d, %v; want 10", n, err)
	}

	if n, err := c.DeleteWhere(ctx, col, Filter{"year": Filter{"$gte": 2040}}); err != nil || n != 10 {
		t.Fatalf("DeleteWhere = %d, %v", n, err)
	}
}

func TestRichFilterAsOfAndHybrid(t *testing.T) {
	ctx := context.Background()
	c := client(t)
	const col = "go_hybrid"
	seedRetrieval(t, c, col)
	q := []float32{1, 0, 0, 0}

	res, err := c.Search(ctx, col, q, SearchOptions{TopK: 50, TemporalBlend: Blend(0),
		Where: Filter{"lang": Filter{"$in": []string{"en"}}, "year": Filter{"$gte": 2010, "$lt": 2020}}})
	if err != nil || fmt.Sprint(hitIDs(res)) != "[r11 r13 r15 r17 r19]" {
		t.Fatalf("rich filter = %v, %v", hitIDs(res), err)
	}

	res, err = c.Search(ctx, col, q, SearchOptions{TopK: 50, AsOf: time.UnixMilli(t0 + 4000)})
	if err != nil || len(res.Hits) != 5 {
		t.Fatalf("as_of = %v, %v", hitIDs(res), err)
	}

	res, err = c.Search(ctx, col, nil, SearchOptions{TopK: 3, Text: "TV-4711"})
	if err != nil || len(res.Hits) != 1 || res.Hits[0].ID != "r17" || res.Hits[0].TextScore <= 0 {
		t.Fatalf("keyword = %+v, %v", res, err)
	}
	res, err = c.Search(ctx, col, q, SearchOptions{TopK: 3, Text: "tv-4711",
		Hybrid: &HybridOptions{Fusion: "weighted", Alpha: 0.2}})
	if err != nil || res.Hits[0].ID != "r17" {
		t.Fatalf("hybrid = %+v, %v", res, err)
	}
	if _, err := c.Search(ctx, col, nil, SearchOptions{}); err == nil {
		t.Fatal("search with neither embedding nor text should fail")
	}
}

func TestAliases(t *testing.T) {
	ctx := context.Background()
	c := client(t)
	for _, n := range []string{"go_kb_v1", "go_kb_v2"} {
		newCollection(t, c, CollectionConfig{Name: n, Dim: 2, NShards: 1})
		if err := c.Upsert(ctx, n, []Vector{{ID: n + "_doc", Embedding: []float32{1, 0}}}); err != nil {
			t.Fatal(err)
		}
	}
	t.Cleanup(func() { _ = c.DeleteAlias(context.Background(), "go_kb") })

	if prev, err := c.SetAlias(ctx, "go_kb", "go_kb_v1"); err != nil || prev != "" {
		t.Fatalf("SetAlias = %q, %v", prev, err)
	}
	res, err := c.Search(ctx, "go_kb", []float32{1, 0}, SearchOptions{TopK: 1})
	if err != nil || res.Hits[0].ID != "go_kb_v1_doc" {
		t.Fatalf("search via alias = %+v, %v", res, err)
	}
	if prev, err := c.SetAlias(ctx, "go_kb", "go_kb_v2"); err != nil || prev != "go_kb_v1" {
		t.Fatalf("swap = %q, %v", prev, err)
	}
	res, err = c.Search(ctx, "go_kb", []float32{1, 0}, SearchOptions{TopK: 1})
	if err != nil || res.Hits[0].ID != "go_kb_v2_doc" {
		t.Fatalf("search after swap = %+v, %v", res, err)
	}
	info, err := c.GetCollection(ctx, "go_kb")
	if err != nil || info.Name != "go_kb_v2" || info.Alias != "go_kb" {
		t.Fatalf("info via alias = %+v, %v", info, err)
	}
	aliases, err := c.ListAliases(ctx)
	if err != nil || aliases["go_kb"] != "go_kb_v2" {
		t.Fatalf("ListAliases = %v, %v", aliases, err)
	}
	if err := c.DeleteAlias(ctx, "go_kb"); err != nil {
		t.Fatal(err)
	}
	if err := c.DeleteAlias(ctx, "go_kb"); !IsNotFound(err) {
		t.Fatalf("second DeleteAlias = %v; want 404", err)
	}
}

func TestCompact(t *testing.T) {
	ctx := context.Background()
	c := client(t)
	const col = "go_compact"
	newCollection(t, c, CollectionConfig{Name: col, Dim: 4, NShards: 2, CompactionRatio: 1000})
	var vecs []Vector
	for i := 0; i < 120; i++ {
		vecs = append(vecs, Vector{ID: fmt.Sprintf("k%03d", i),
			Embedding: []float32{1, float32(i % 9), float32(i % 13), float32(i)},
			Payload:   map[string]string{"keep": fmt.Sprint(i%4 == 0)}})
	}
	if err := c.Upsert(ctx, col, vecs); err != nil {
		t.Fatal(err)
	}
	if n, err := c.DeleteWhere(ctx, col, Filter{"keep": "false"}); err != nil || n != 90 {
		t.Fatalf("DeleteWhere = %d, %v", n, err)
	}
	before, err := c.CompactionStatus(ctx, col)
	if err != nil || before.Tombstones != 90 || before.Running {
		t.Fatalf("status = %+v, %v", before, err)
	}
	after, err := c.Compact(ctx, col, true)
	if err != nil || after.Started != 2 || after.Running || after.Tombstones != 0 ||
		after.LastReclaimed != 90 || after.IndexBytes >= before.IndexBytes {
		t.Fatalf("compact = %+v, %v", after, err)
	}
	if n, err := c.CountWhere(ctx, col, Filter{"keep": "true"}, time.Time{}); err != nil || n != 30 {
		t.Fatalf("count after compact = %d, %v", n, err)
	}
	if _, err := c.Compact(ctx, "go_missing", false); err == nil {
		t.Fatal("compact of a missing collection should fail")
	}
}
