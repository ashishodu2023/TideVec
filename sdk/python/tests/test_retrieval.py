"""Get-by-id, scroll, rich filters, as_of, keyword / hybrid search and
aliases — run against both the embedded engine and a live server."""

import datetime as dt

import pytest

from tidevec import CollectionNotFoundError, Record, TideVecError

T0 = 1_700_000_000_000          # 2023-11-14T22:13:20Z
DIM = 4


def _vec(i: int):
    return [1.0, i / 50.0, 0.0, 0.0]


@pytest.fixture
def docs(db, name):
    db.create_collection(name, dim=DIM, n_shards=2, temporal_blend=0.0, text_field="text")
    vectors = []
    for i in range(50):
        text = f"plain note {i}"
        if i == 17:
            text = "error code TV-4711 on shutdown"
        vectors.append({
            "id": f"r{i:02d}",
            "embedding": _vec(i),
            "created_at": T0 + i * 1000,
            "payload": {"lang": "en" if i % 2 else "de", "year": 2000 + i,
                        "draft": i % 5 == 0, "text": text},
        })
    db.upsert(name, vectors)
    return name


def test_collection_info_reports_text_field(db, docs):
    info = db.get_collection(docs)
    assert info.text_field == "text"
    assert info.payload_index is True
    assert info.dim == DIM


def test_get_and_get_many(db, docs):
    rec = db.get(docs, "r07")
    assert isinstance(rec, Record)
    assert rec.payload["lang"] == "en" and rec.payload["year"] == "2007"
    assert rec.created_at == T0 + 7000
    assert rec.embedding == pytest.approx(_vec(7))
    assert db.get(docs, "nope") is None

    recs = db.get_many(docs, ["r03", "missing", "r01"])
    assert [r.id for r in recs] == ["r03", "r01"]
    assert recs[0].embedding is None
    assert db.get_many(docs, ["r03"], with_payload=False)[0].payload == {}


def test_get_reports_supersession(db, name):
    db.create_collection(name, dim=DIM, n_shards=2)
    db.upsert(name, [{"id": "old", "embedding": [1, 0, 0, 0]},
                     {"id": "new", "embedding": [0, 1, 0, 0]}])
    db.add_edges(name, [{"src": "new", "tgt": "old", "type": "UPDATES"}])
    old = db.get(name, "old", with_vector=False)
    assert old.superseded_by == ["new"] and old.is_superseded
    assert db.get(name, "new").supersedes == ["old"]


def test_scroll_pages_in_id_order(db, docs):
    seen, cursor, pages = [], None, 0
    while True:
        page = db.scroll(docs, limit=16, cursor=cursor)
        seen += [r.id for r in page]
        pages += 1
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    assert pages == 4
    assert seen == sorted(seen) and len(seen) == 50

    assert len(list(db.iterate(docs, batch_size=7))) == 50
    en = list(db.iterate(docs, filter={"lang": "en"}, with_vector=True))
    assert len(en) == 25 and all(r.embedding is not None for r in en)


def test_list_ids_and_count(db, docs):
    ids = db.list_ids(docs)
    assert len(ids) == 50 and ids[0] == "r00"
    assert db.list_ids(docs, limit=3, cursor="r10") == ["r11", "r12", "r13"]
    assert db.list_ids(docs, filter="lang=de", limit=3) == ["r00", "r02", "r04"]
    assert db.count(docs) == 50
    assert db.count(docs, filter={"$or": [{"lang": "de"}, {"year": {"$gte": 2045}}]}) == 28
    assert db.count(docs, filter={"draft": True}) == 10
    assert db.count(docs, as_of=T0 + 9000) == 10


@pytest.mark.parametrize("flt,expected", [
    ({"lang": "en", "year": {"$gte": 2010, "$lt": 2020}}, {11, 13, 15, 17, 19}),
    ({"lang": {"$in": ["de"]}, "year": {"$lte": 2004}}, {0, 2, 4}),
    ({"year": {"$gt": 2046}}, {47, 48, 49}),
    ({"$and": [{"lang": "de"}, {"$not": {"year": {"$gte": 2004}}}]}, {0, 2}),
    ({"lang": {"$ne": "en"}, "year": {"$nin": [2000, 2002, 2004]},
      "_id": {"$in": ["r00", "r06", "r08", "r10", "r11"]}}, {6, 8, 10}),
    ({"created_at": {"$lte": "2023-11-14T22:13:23Z"}}, {0, 1, 2, 3}),
    ({"created_at": {"$lt": dt.datetime(2023, 11, 14, 22, 13, 22)}}, {0, 1}),
    ({"missing_key": {"$exists": False}, "year": 2003}, {3}),
    ("lang=en", set(range(1, 50, 2))),
], ids=["and_range", "in_lte", "gt", "nested_not", "ne_nin_id", "created_at_iso",
        "created_at_datetime", "exists", "legacy_string"])
def test_rich_filters(db, docs, flt, expected):
    res = db.search(docs, [1, 0, 0, 0], top_k=50, filter=flt)
    assert {int(h.id[1:]) for h in res} == expected


def test_bad_filter_is_rejected(db, docs):
    with pytest.raises((TideVecError, ValueError)):
        db.search(docs, [1, 0, 0, 0], filter={"lang": {"$regex": "e"}})
    with pytest.raises(ValueError):
        db.search(docs, [1, 0, 0, 0], filter="no-equals-sign")


def test_as_of_search(db, docs):
    res = db.search(docs, [1, 0, 0, 0], top_k=50, as_of=T0 + 4000)
    assert sorted(h.id for h in res) == ["r00", "r01", "r02", "r03", "r04"]
    as_dt = dt.datetime.fromtimestamp((T0 + 4000) / 1000, tz=dt.timezone.utc)
    assert len(db.search(docs, [1, 0, 0, 0], top_k=50, as_of=as_dt)) == 5
    assert len(db.search(docs, [1, 0, 0, 0], top_k=50,
                         as_of="2023-11-14T22:13:24Z")) == 5


def test_as_of_supersession(db, name):
    db.create_collection(name, dim=DIM, n_shards=2, temporal_blend=0.0,
                         supersession=True, supersede_threshold=0.9,
                         supersede_match_field="doc")
    db.upsert(name, [
        {"id": "v1", "embedding": [1, 0, 0, 0], "created_at": T0, "payload": {"doc": "a"}},
        {"id": "v2", "embedding": [1, 0.1, 0, 0], "created_at": T0 + 10_000,
         "payload": {"doc": "a"}},
    ])
    now = {h.id: h for h in db.search(name, [1, 0, 0, 0], top_k=2)}
    assert now["v1"].superseded_by == ["v2"]
    then = {h.id: h for h in db.search(name, [1, 0, 0, 0], top_k=2, as_of=T0 + 5000)}
    assert set(then) == {"v1"} and then["v1"].superseded_by == []


def test_keyword_and_hybrid_search(db, docs):
    res = db.search(docs, text="TV-4711", top_k=3)
    assert [h.id for h in res] == ["r17"] and res[0].text_score > 0

    res = db.search(docs, [1, 0, 0, 0], text="tv-4711 shutdown", top_k=3)
    assert "r17" in [h.id for h in res]
    res = db.search(docs, [1, 0, 0, 0], text="tv-4711", top_k=3, hybrid="weighted", alpha=0.2)
    assert res[0].id == "r17"
    assert len(db.search(docs, text="tv-4711", filter={"lang": "de"})) == 0

    with pytest.raises(ValueError):
        db.search(docs)
    with pytest.raises(ValueError):
        db.search(docs, [1, 0, 0, 0], hybrid="rrf")


def test_keyword_search_needs_text_field(db, name):
    db.create_collection(name, dim=DIM)
    db.upsert(name, [{"id": "a", "embedding": [1, 0, 0, 0], "payload": {"text": "hello"}}])
    with pytest.raises((TideVecError, ValueError)):
        db.search(name, text="hello")


def test_delete_by_filter_dict(db, docs):
    assert db.delete(docs, filter={"year": {"$gte": 2040}}) == 10
    assert db.count(docs) == 40
    assert db.delete(docs, filter="lang=en") == 20
    assert db.count(docs) == 20


def test_aliases(db, name):
    v1, v2, alias = name + "_v1", name + "_v2", name + "_live"
    for n in (v1, v2):
        db.create_collection(n, dim=DIM)
        db.upsert(n, [{"id": n + "_doc", "embedding": [1, 0, 0, 0]}])

    assert db.set_alias(alias, v1) is None
    assert db.search(alias, [1, 0, 0, 0])[0].id == v1 + "_doc"
    info = db.get_collection(alias)
    assert info.name == v1 and alias in info.aliases

    assert db.set_alias(alias, v2) == v1          # atomic swap
    assert db.search(alias, [1, 0, 0, 0])[0].id == v2 + "_doc"
    assert db.get(alias, v2 + "_doc") is not None
    db.upsert(alias, [{"id": "via_alias", "embedding": [0, 1, 0, 0]}])
    assert db.count(v2) == 2
    assert db.list_aliases().get(alias) == v2

    with pytest.raises((TideVecError, ValueError)):
        db.set_alias(v1, v2)                      # can't shadow a collection
    with pytest.raises((TideVecError, CollectionNotFoundError)):
        db.set_alias(alias, name + "_nope")

    assert db.delete_alias(alias) is True
    assert db.delete_alias(alias) is False
    assert alias not in db.list_aliases()
    with pytest.raises(CollectionNotFoundError):
        db.search(alias, [1, 0, 0, 0])


def test_compact(db, name):
    db.create_collection(name, dim=DIM, n_shards=2, compaction_ratio=1000.0)
    db.upsert(name, [{"id": f"k{i:03d}", "embedding": [1, i % 9, i % 13, i],
                      "payload": {"keep": str(i % 4 == 0).lower()}} for i in range(120)])
    assert db.delete(name, filter={"keep": "false"}) == 90

    before = db.compaction_status(name)
    assert before.tombstones == 90 and not before.running
    after = db.compact(name, wait=True)
    assert after.started == 2 and not after.running
    assert after.tombstones == 0 and after.last_reclaimed == 90 and after.runs == 2
    assert after.index_bytes < before.index_bytes
    assert db.count(name, filter={"keep": "true"}) == 30
    hits = db.search(name, [1, 0, 0, 0], top_k=5, filter={"keep": "true"})
    assert len(hits) == 5 and all(h.payload["keep"] == "true" for h in hits)
    assert db.compact(name).started == 0
    with pytest.raises(TideVecError):
        db.create_collection(name + "_bad", dim=DIM, compaction_ratio=0)
