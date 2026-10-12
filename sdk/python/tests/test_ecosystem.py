"""
Framework adapters: Haystack, DSPy, LangGraph, CrewAI.

Each block skips when its framework is not installed. Install them with
``pip install "tidevec[haystack,dspy,langgraph,crewai]"`` to run everything.
"""

import importlib
import math
import os
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest

# Keep framework imports from phoning home or writing to $HOME.
os.environ.setdefault("HAYSTACK_TELEMETRY_ENABLED", "False")
os.environ.setdefault("DSPY_CACHEDIR", os.path.join(tempfile.gettempdir(), "tidevec-dspy-cache"))
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from tidevec.embeddings import HashingEmbedder  # noqa: E402
from tidevec.integrations import _backend  # noqa: E402
from tidevec.integrations._backend import Record, legacy_pushdown, matches  # noqa: E402


def _optional(module: str):
    try:
        return importlib.import_module(module)
    except Exception:  # not installed, or cannot initialise (e.g. read-only config dir)
        return None


def _require(*modules: str):
    """Skip the test unless every module imports; returns the first."""
    loaded = []
    for module in modules:
        try:
            loaded.append(pytest.importorskip(module, reason=f"{module} not installed"))
        except Exception as e:  # pragma: no cover - installed but cannot initialise
            pytest.skip(f"{module} failed to import: {e}")
    return loaded[0]


@pytest.fixture
def server(server_url):
    from tidevec import TideVec
    client = TideVec(server_url)
    before = {c.name for c in client.list_collections()}
    yield client
    for c in client.list_collections():
        if c.name not in before:
            client.drop_collection(c.name)


@pytest.fixture(params=["server", "local"])
def anydb(request):
    """A live server and embedded mode (skipped without tidevec_native)."""
    yield request.getfixturevalue("server" if request.param == "server" else "local_db")


def _uniq(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


# ---- shared filter evaluator (no framework needed) ----------------------

def _rec(payload, created_at=1_700_000_000_000):
    return Record(id="x", created_at=created_at, payload=payload)


def test_filter_evaluator_matches_server_semantics():
    r = _rec({"lang": "en", "year": "2021", "tag": "a", "when": "2024-05-01T00:00:00Z"})
    assert matches({"lang": "en"}, r)
    assert not matches({"lang": "de"}, r)
    assert matches({"year": 2021}, r)
    assert matches({"year": {"$gte": 2020, "$lt": 2022}}, r)
    assert not matches({"year": {"$gt": 2021}}, r)
    assert matches({"tag": ["a", "b"]}, r)
    assert matches({"tag": {"$nin": ["c"]}}, r)
    assert matches({"missing": {"$exists": False}}, r)
    assert matches({"$or": [{"lang": "de"}, {"tag": "a"}]}, r)
    assert matches({"$not": {"lang": "de"}}, r)
    assert matches({"when": {"$gte": "2024-01-01"}}, r)
    assert matches({"created_at": {"$lt": 1_800_000_000_000}}, r)
    with pytest.raises(ValueError):
        matches({"lang": {"$regex": "e"}}, r)


def test_legacy_pushdown_is_a_superset():
    assert legacy_pushdown({"a": "1", "b": {"$gt": 2}}) == "a=1"
    assert legacy_pushdown({"$and": [{"b": {"$gt": 2}}, {"c": 3}]}) == "c=3"
    assert legacy_pushdown({"$or": [{"a": "1"}, {"a": "2"}]}) == ""
    assert legacy_pushdown({"created_at": 5}) == ""


@pytest.mark.parametrize("structured", [True, False])
def test_backend_paths_agree(anydb, structured):
    name = _uniq("eco_backend")
    anydb.create_collection(name, dim=4)
    now = int(time.time() * 1000)
    anydb.upsert(name, [
        {"id": f"v{i}", "embedding": [1.0, i / 10, 0.0, 0.0],
         "payload": {"grp": "odd" if i % 2 else "even", "n": str(i)},
         "created_at": now - i * 1000}
        for i in range(10)])
    b = _backend.Backend(anydb, name, structured=structured)
    assert b.count() == 10
    assert b.count({"grp": "odd", "n": {"$gte": 5}}) == 3
    assert sorted(r.id for r in b.scan({"n": {"$in": ["1", "2"]}})) == ["v1", "v2"]
    assert [r.id for r in b.get(["v3", "nope", "v1"])] == ["v3", "v1"]
    hits = b.search([1.0, 0.0, 0.0, 0.0], 3, {"grp": "even"}, temporal_blend=0.0)
    assert [h.id for h in hits] == ["v0", "v2", "v4"]
    assert b.delete(flt={"grp": "odd", "n": {"$lt": 4}}) == 2
    assert b.count() == 8


def test_backend_probe_detects_structured_api(anydb):
    name = _uniq("eco_probe")
    anydb.create_collection(name, dim=4)
    assert _backend.Backend(anydb, name).structured is True


# ---- Haystack ------------------------------------------------------------

haystack = _optional("haystack")

if haystack is not None:
    from haystack import Document, Pipeline
    from haystack.document_stores.errors import DuplicateDocumentError
    from haystack.document_stores.types import DuplicatePolicy
    from haystack.testing.document_store import (
        CountDocumentsTest,
        DeleteDocumentsTest,
        FilterDocumentsTest,
        WriteDocumentsTest,
    )

    from tidevec.integrations.haystack import TideVecDocumentStore, TideVecEmbeddingRetriever

    def _same_embedding(a, b):
        if a is None or b is None:
            return a is b
        return len(a) == len(b) and all(math.isclose(x, y, rel_tol=1e-5, abs_tol=1e-6)
                                        for x, y in zip(a, b))

    class TestHaystackConformance(CountDocumentsTest, DeleteDocumentsTest,
                                  FilterDocumentsTest, WriteDocumentsTest):
        """Haystack's own DocumentStore test-suites, run against TideVec."""

        @pytest.fixture
        def document_store(self, server):
            return TideVecDocumentStore(server, collection=_uniq("hs"), embedding_dim=768)

        def assert_documents_are_equal(self, received, expected):
            # Embeddings round-trip through float32.
            key = lambda d: d.id  # noqa: E731
            received, expected = sorted(received, key=key), sorted(expected, key=key)
            assert [d.id for d in received] == [d.id for d in expected]
            for r, e in zip(received, expected):
                assert r.content == e.content
                assert r.meta == e.meta
                assert _same_embedding(r.embedding, e.embedding)

        def test_write_documents(self, document_store):
            docs = [Document(id="1")]
            assert document_store.write_documents(docs) == 1
            with pytest.raises(DuplicateDocumentError):
                document_store.write_documents(docs, DuplicatePolicy.FAIL)

    @pytest.fixture
    def hs_store(anydb):
        return TideVecDocumentStore(anydb, collection=_uniq("hs_rag"), embedding_dim=384,
                                    supersede_match_field="doc_id")

    def test_haystack_retriever_drops_superseded_versions(hs_store):
        emb = HashingEmbedder(384)
        old_t = int(time.time() * 1000) - 86_400_000
        hs_store.write_documents([
            Document(id="refund-v1", content="Refunds are processed within 14 days",
                     embedding=emb.embed_query("Refunds are processed within 14 days"),
                     meta={"doc_id": "refunds", "created_at": old_t}),
            Document(id="ship", content="We ship worldwide with tracking",
                     embedding=emb.embed_query("We ship worldwide with tracking"),
                     meta={"doc_id": "shipping"}),
        ])
        hs_store.write_documents([
            Document(id="refund-v2", content="Refunds are processed within 3 days",
                     embedding=emb.embed_query("Refunds are processed within 3 days"),
                     meta={"doc_id": "refunds"}),
        ])

        pipe = Pipeline()
        pipe.add_component("retriever", TideVecEmbeddingRetriever(document_store=hs_store, top_k=3))
        out = pipe.run({"retriever": {"query_embedding": emb.embed_query("how long do refunds take")}})
        docs = out["retriever"]["documents"]
        ids = [d.id for d in docs]
        assert "refund-v2" in ids and "refund-v1" not in ids
        assert docs[0].meta["tidevec"]["superseded_by"] == []

        all_versions = hs_store.embedding_retrieval(
            emb.embed_query("how long do refunds take"), top_k=3, exclude_superseded=False)
        old = next(d for d in all_versions if d.id == "refund-v1")
        assert old.meta["tidevec"]["superseded_by"] == ["refund-v2"]

        filtered = hs_store.embedding_retrieval(
            emb.embed_query("shipping"), top_k=3,
            filters={"field": "meta.doc_id", "operator": "==", "value": "shipping"})
        assert [d.id for d in filtered] == ["ship"]

    def test_haystack_serialization_roundtrip(hs_store):
        retriever = TideVecEmbeddingRetriever(document_store=hs_store, top_k=7)
        data = retriever.to_dict()
        assert "api_key" not in str(data)
        clone = TideVecEmbeddingRetriever.from_dict(data)
        assert clone.top_k == 7
        assert clone.document_store.collection == hs_store.collection
        assert clone.document_store.supersede_match_field == "doc_id"


# ---- DSPy ----------------------------------------------------------------

def test_dspy_retriever(anydb):
    dspy = _require("dspy")
    from tidevec.integrations.dspy import TideVecRetriever

    emb = HashingEmbedder(256)
    name = _uniq("dspy")
    retrieve = TideVecRetriever(anydb, name, embedder=emb, k=2, supersession=True,
                                supersede_match_field="doc_id", supersede_threshold=0.1)
    old_t = int(time.time() * 1000) - 3_600_000
    retrieve.add(["The office is in Berlin", "Lunch is served at noon"],
                 metadatas=[{"doc_id": "office"}, {"doc_id": "lunch"}],
                 ids=["office-v1", "lunch"], created_at=old_t)
    retrieve.add(["The office moved to Munich"], metadatas=[{"doc_id": "office"}],
                 ids=["office-v2"])

    pred = retrieve("where is the office")
    assert isinstance(pred, dspy.Prediction)
    assert pred.ids[0] == "office-v2"
    assert "office-v1" not in pred.ids
    assert pred.metadata[0]["doc_id"] == "office"
    assert len(pred.passages) == len(pred.scores) <= 2

    multi = retrieve(["where is the office", "when is lunch"], k=3)
    assert set(multi.ids) == {"office-v2", "lunch"}

    assert retrieve("office", filter="doc_id=lunch").ids == ["lunch"]

    # Legacy rm= path used by dspy.Retrieve.
    with dspy.context(rm=retrieve.as_rm()):
        passages = dspy.Retrieve(k=1)("lunch time").passages
    assert passages == ["Lunch is served at noon"]


def test_dspy_retriever_accepts_plain_callable(anydb):
    _require("dspy")
    from tidevec.integrations.dspy import TideVecRetriever

    emb = HashingEmbedder(64)
    retrieve = TideVecRetriever(anydb, _uniq("dspy_fn"), embedder=emb.embed_documents, k=1)
    retrieve.add(["alpha beta", "gamma delta"])
    assert retrieve("gamma delta").passages == ["gamma delta"]


# ---- LangGraph -----------------------------------------------------------

@pytest.fixture
def lg_store(anydb):
    _require("langgraph.store.base")
    from tidevec.integrations.langgraph import TideVecStore

    emb = HashingEmbedder(128)
    return TideVecStore(anydb, collection=_uniq("lg"),
                        index={"dims": 128, "embed": emb.embed_documents, "fields": ["text"]})


def test_langgraph_store_crud_and_search(lg_store):
    ns = ("users", "u1", "memories")
    lg_store.put(ns, "diet", {"text": "User is vegetarian", "kind": "pref", "level": 2})
    lg_store.put(ns, "city", {"text": "User lives in Lisbon", "kind": "fact", "level": 5})
    lg_store.put(("users", "u2", "memories"), "diet", {"text": "User eats fish", "kind": "pref"})

    item = lg_store.get(ns, "diet")
    assert item.value["text"] == "User is vegetarian"
    assert item.namespace == ns and item.key == "diet"
    created = item.created_at

    time.sleep(0.01)
    lg_store.put(ns, "diet", {"text": "User is vegan", "kind": "pref", "level": 3})
    updated = lg_store.get(ns, "diet")
    assert updated.value["text"] == "User is vegan"
    assert updated.created_at == created and updated.updated_at > created

    hits = lg_store.search(("users", "u1"), query="what does the user eat", limit=1)
    assert [h.key for h in hits] == ["diet"] and hits[0].score is not None

    assert {h.key for h in lg_store.search(("users",), filter={"kind": "pref"})} == {"diet"}
    assert len(lg_store.search(("users",), filter={"kind": "pref"}, limit=10)) == 2
    assert [h.key for h in lg_store.search(ns, filter={"level": {"$gte": 4}})] == ["city"]
    assert lg_store.search(("nobody",)) == []

    assert lg_store.list_namespaces(prefix=("users",)) == [
        ("users", "u1", "memories"), ("users", "u2", "memories")]
    assert lg_store.list_namespaces(max_depth=2) == [("users", "u1"), ("users", "u2")]
    assert lg_store.list_namespaces(suffix=("memories",), prefix=("users", "*")) == [
        ("users", "u1", "memories"), ("users", "u2", "memories")]

    lg_store.delete(ns, "diet")
    assert lg_store.get(ns, "diet") is None
    assert [h.key for h in lg_store.search(ns)] == ["city"]


def test_langgraph_store_unindexed_items_and_batch(lg_store):
    from langgraph.store.base import GetOp, PutOp, SearchOp

    ns = ("docs",)
    results = lg_store.batch([
        PutOp(ns, "a", {"text": "apples are red"}),
        PutOp(ns, "b", {"text": "bananas are yellow"}, index=False),
        GetOp(ns, "a"),
        SearchOp(ns, query="red fruit", limit=5),
    ])
    assert results[0] is None and results[1] is None
    assert results[2].value == {"text": "apples are red"}
    keys = [i.key for i in results[3]]
    assert keys[0] == "a" and "b" in keys
    assert next(i for i in results[3] if i.key == "b").score is None


def test_langgraph_store_async_and_ttl(anydb):
    _require("langgraph.store.base")
    import asyncio

    from tidevec.integrations.langgraph import TideVecStore

    store = TideVecStore(anydb, collection=_uniq("lg_ttl"))

    async def run():
        await store.aput(("s",), "k", {"v": 1}, ttl=1 / 60)  # one second
        assert (await store.aget(("s",), "k")).value == {"v": 1}

    asyncio.run(run())
    time.sleep(1.2)
    assert store.get(("s",), "k") is None


def test_langgraph_memory_tools(anydb):
    _require("langgraph.store.base", "langchain_core.tools")
    from tidevec.integrations.langgraph import memory_tools
    from tidevec.memory import AgentMemory

    mem = AgentMemory(anydb, _uniq("lg_mem"), embedder=HashingEmbedder(128))
    remember, recall, history = memory_tools(mem, namespace_key="u1")
    remember.invoke({"text": "User prefers tea", "key": "drink"})
    out = remember.invoke({"text": "User prefers coffee now", "key": "drink"})
    assert "replaced 1" in out
    assert "coffee" in recall.invoke({"query": "what does the user drink"})
    assert "tea" not in recall.invoke({"query": "what does the user drink"})
    hist = history.invoke({"key": "drink"})
    assert hist.index("tea") < hist.index("coffee") and "[superseded]" in hist


# ---- CrewAI --------------------------------------------------------------

@pytest.fixture
def crew_storage(anydb):
    _require("crewai.memory.types")
    from tidevec.integrations.crewai import TideVecStorage
    return TideVecStorage(anydb, collection=_uniq("crew"))


def _crew_record(text, emb, **kw):
    from crewai.memory.types import MemoryRecord
    return MemoryRecord(content=text, embedding=emb.embed_query(text), **kw)


def test_crewai_storage_protocol(crew_storage):
    from crewai.memory.storage.backend import StorageBackend

    assert isinstance(crew_storage, StorageBackend)
    emb = HashingEmbedder(96)
    old = datetime.now(timezone.utc) - timedelta(days=30)
    recs = [
        _crew_record("Alpha launch is in March", emb, scope="/project/alpha",
                     categories=["planning"], metadata={"owner": "ana"}, created_at=old),
        _crew_record("Alpha budget is 40k", emb, scope="/project/alpha/finance",
                     categories=["finance"], metadata={"owner": "bo"}),
        _crew_record("Beta uses Postgres", emb, scope="/project/beta",
                     categories=["tech"], importance=0.9),
        _crew_record("Office closes at 6pm", emb, scope="/company"),
    ]
    assert crew_storage.count() == 0 and crew_storage.search(emb.embed_query("x")) == []
    crew_storage.save(recs)

    assert crew_storage.count() == 4
    assert crew_storage.count("/project") == 3
    assert crew_storage.count("/project/alpha") == 2
    assert crew_storage.count("/proj") == 0  # path prefix, not string prefix

    got = crew_storage.get_record(recs[2].id)
    assert got.content == "Beta uses Postgres" and got.importance == 0.9
    assert got.scope == "/project/beta" and got.embedding is not None

    hits = crew_storage.search(emb.embed_query("when is the alpha launch"), scope_prefix="/project",
                               limit=2)
    assert hits[0][0].id == recs[0].id and 0 < hits[0][1] <= 1
    assert all(r.scope.startswith("/project") for r, _ in hits)
    assert [r.id for r, _ in crew_storage.search(emb.embed_query("alpha"), categories=["finance"])] \
        == [recs[1].id]
    assert [r.id for r, _ in crew_storage.search(emb.embed_query("alpha"),
                                                 metadata_filter={"owner": "ana"})] == [recs[0].id]
    assert crew_storage.search(emb.embed_query("alpha"), min_score=1.01) == []

    listed = crew_storage.list_records("/project")
    assert [r.id for r in listed][-1] == recs[0].id  # newest first
    assert len(crew_storage.list_records(limit=2, offset=1)) == 2

    info = crew_storage.get_scope_info("/project")
    assert info.record_count == 3
    assert info.child_scopes == ["/project/alpha", "/project/beta"]
    assert info.categories == ["finance", "planning", "tech"]
    assert info.oldest_record < info.newest_record
    assert crew_storage.list_scopes("/") == ["/company", "/project"]
    assert crew_storage.list_categories("/project/alpha") == {"planning": 1, "finance": 1}

    updated = got.model_copy(update={"content": "Beta uses Postgres 17", "embedding": None})
    crew_storage.update(updated)
    again = crew_storage.get_record(got.id)
    assert again.content == "Beta uses Postgres 17" and again.embedding is not None

    assert crew_storage.delete(older_than=datetime.now(timezone.utc) - timedelta(days=1)) == 1
    assert crew_storage.delete(scope_prefix="/project", categories=["finance"]) == 1
    assert crew_storage.delete(record_ids=[recs[3].id]) == 1
    assert crew_storage.count() == 1
    crew_storage.reset("/project")
    assert crew_storage.count() == 0
    crew_storage.reset()
    assert crew_storage.count() == 0


def test_crewai_unified_memory_shallow_recall(crew_storage):
    import asyncio

    from crewai import Memory

    emb = HashingEmbedder(96)
    memory = Memory(storage=crew_storage, embedder=emb.embed_documents, llm="openai/gpt-4o-mini")
    # Pre-classified records skip the LLM analysis step.
    crew_storage.save([
        _crew_record("The deploy window is Tuesday 10am", emb, scope="/ops", categories=["ops"]),
        _crew_record("Coffee machine is on floor 3", emb, scope="/office"),
    ])
    matches_ = memory.recall("what is the deploy window", depth="shallow", limit=1)
    assert matches_ and matches_[0].record.content.startswith("The deploy window")

    asyncio.run(crew_storage.asave([_crew_record("Async works", emb)]))
    assert asyncio.run(crew_storage.adelete(scope_prefix="/office")) == 1
