import time

import pytest

from tidevec import CollectionNotFoundError, HalfLife, TideVecError

DAY = 86_400_000


def now_ms() -> int:
    return int(time.time() * 1000)


def test_create_upsert_search(db, name):
    db.create_collection(name, dim=3, n_shards=2)
    assert db.has_collection(name)
    db.upsert(name, [
        {"id": "a", "embedding": [1, 0, 0], "payload": {"lang": "en"}},
        {"id": "b", "embedding": [0, 1, 0], "payload": {"lang": "de"}},
        {"id": "c", "embedding": [0.9, 0.1, 0], "payload": {"lang": "en", "page": 3}},
    ])
    assert db.count(name) == 3

    hits = db.search(name, [1, 0, 0], top_k=2, temporal_blend=0.0)
    assert [h.id for h in hits] == ["a", "c"]
    assert hits[0].score == pytest.approx(hits[0].vector_score, abs=1e-5)
    assert hits[1].payload["page"] == "3"

    only_de = db.search(name, [1, 0, 0], top_k=3, filter="lang=de")
    assert [h.id for h in only_de] == ["b"]


def test_missing_collection_raises(db):
    assert not db.has_collection("does_not_exist")
    with pytest.raises(CollectionNotFoundError):
        db.search("does_not_exist", [1.0, 0.0])


def test_dimension_mismatch_is_rejected(db, name):
    db.create_collection(name, dim=3)
    with pytest.raises(TideVecError):
        db.upsert(name, [{"id": "x", "embedding": [1, 0]}])
    with pytest.raises(TideVecError):
        db.search(name, [1, 0])
    assert db.count(name) == 0


def test_delete_by_ids_and_filter(db, name):
    db.create_collection(name, dim=2, n_shards=2)
    db.upsert(name, [{"id": f"v{i}", "embedding": [1, i], "payload": {"doc": "a" if i < 3 else "b"}}
                     for i in range(6)])
    assert db.delete(name, filter="doc=a") == 3
    assert db.delete(name, ["v3", "missing"]) == 1
    assert db.count(name) == 2
    with pytest.raises(ValueError):
        db.delete(name)
    with pytest.raises(ValueError):
        db.delete(name, filter="no-equals-sign")


def test_per_query_blend_overrides_collection(db, name):
    db.create_collection(name, dim=2, half_life_ms=HalfLife.ONE_HOUR, temporal_blend=0.9)
    db.upsert(name, [
        {"id": "old_exact", "embedding": [1, 0], "created_at": now_ms() - 30 * DAY},
        {"id": "fresh_near", "embedding": [0.6, 0.8]},
    ])
    assert db.search(name, [1, 0], top_k=1)[0].id == "fresh_near"
    assert db.search(name, [1, 0], top_k=1, temporal_blend=0.0)[0].id == "old_exact"


def test_supersession_marks_older_version(db, name):
    db.create_collection(name, dim=4, n_shards=2, supersession=True,
                         supersede_threshold=0.9, supersede_match_field="doc_id")
    db.upsert(name, [
        {"id": "refund_v1", "embedding": [1, 0, 0, 0], "created_at": now_ms() - 30 * DAY,
         "payload": {"doc_id": "refunds"}},
        {"id": "other_doc", "embedding": [0.99, 0.1, 0, 0], "created_at": now_ms() - 20 * DAY,
         "payload": {"doc_id": "shipping"}},
    ])
    db.upsert(name, [{"id": "refund_v2", "embedding": [0.97, 0.2, 0, 0],
                      "payload": {"doc_id": "refunds"}}])

    hits = {h.id: h for h in db.search(name, [1, 0, 0, 0], top_k=3, temporal_blend=0.0)}
    assert hits["refund_v1"].superseded_by == ["refund_v2"]
    assert hits["refund_v1"].is_superseded
    assert hits["other_doc"].superseded_by == []

    ranked = db.search(name, [1, 0, 0, 0], top_k=3, temporal_blend=0.0, superseded_penalty=1.0)
    assert ranked[-1].id == "refund_v1"


def test_edges_and_contradiction_mode(db, name):
    db.create_collection(name, dim=2)
    db.upsert(name, [{"id": "claim", "embedding": [1, 0]}, {"id": "rebuttal", "embedding": [0, 1]}])
    assert db.add_edges(name, [{"src": "rebuttal", "tgt": "claim", "type": "CONTRADICTS"}]) == 1
    hit = db.search(name, [1, 0], top_k=1, mode="contradiction_check")[0]
    assert hit.id == "claim"
    assert "rebuttal" in hit.contradicted_by


def test_numpy_embeddings_accepted(db, name):
    np = pytest.importorskip("numpy")
    db.create_collection(name, dim=3)
    db.upsert(name, [{"id": "n", "embedding": np.array([0.0, 1.0, 0.0], dtype=np.float32)}])
    assert db.search(name, np.array([0, 1, 0]), top_k=1)[0].id == "n"


def test_local_persists_across_reopen(tmp_path):
    pytest.importorskip("tidevec_native")
    from tidevec import Local
    path = str(tmp_path / "persist")
    with Local(path) as db:
        db.create_collection("p", dim=2, supersession=True, supersede_threshold=0.9)
        db.upsert("p", [{"id": "old", "embedding": [1, 0], "created_at": now_ms() - DAY}])
        db.upsert("p", [{"id": "new", "embedding": [0.99, 0.05]}])
    db = Local(path)
    assert db.count("p") == 2
    hits = {h.id: h for h in db.search("p", [1, 0], top_k=2)}
    assert hits["old"].superseded_by == ["new"]
    db.close()
