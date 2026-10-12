import time

import pytest

from tidevec.embeddings import HashingEmbedder

DAY = 86_400_000


def now_ms() -> int:
    return int(time.time() * 1000)


# ---- LangChain ------------------------------------------------------

def test_langchain_roundtrip_and_supersession(db, name):
    pytest.importorskip("langchain_core")
    from tidevec.integrations.langchain import TideVecVectorStore

    store = TideVecVectorStore(db, name, HashingEmbedder(256), supersession=True,
                               supersede_threshold=0.5, supersede_match_field="doc_id")
    old_ids = store.add_texts(["Refunds are processed within 14 days of the request"],
                              metadatas=[{"doc_id": "refunds", "version": 1, "tags": ["policy"]}],
                              created_at=now_ms() - 60 * DAY)
    new_ids = store.add_texts(["Refunds are processed within 3 days of the request"],
                              metadatas=[{"doc_id": "refunds", "version": 2, "tags": ["policy"]}])
    store.add_texts(["Shipping is free above 50 euros"], metadatas=[{"doc_id": "shipping"}])

    docs = store.similarity_search("how long do refunds take", k=3)
    by_id = {d.id: d for d in docs}
    assert by_id[old_ids[0]].metadata["tidevec"]["superseded_by"] == new_ids
    assert by_id[new_ids[0]].metadata["version"] == 2
    assert by_id[new_ids[0]].metadata["tags"] == ["policy"]

    current = store.similarity_search("how long do refunds take", k=2, exclude_superseded=True)
    assert old_ids[0] not in [d.id for d in current]

    filtered = store.similarity_search("refunds", k=5, filter={"doc_id": "shipping"})
    assert [d.page_content for d in filtered] == ["Shipping is free above 50 euros"]

    scored = store.similarity_search_with_relevance_scores("refunds", k=1)
    assert 0.0 <= scored[0][1] <= 1.0

    assert store.delete(filter={"doc_id": "refunds"})
    assert [d.metadata["doc_id"] for d in store.similarity_search("refunds", k=5)] == ["shipping"]


def test_langchain_from_texts_uses_embedded_mode(tmp_path):
    pytest.importorskip("langchain_core")
    pytest.importorskip("tidevec_native")
    from tidevec.integrations.langchain import TideVecVectorStore

    store = TideVecVectorStore.from_texts(["alpha", "beta"], HashingEmbedder(64),
                                          path=str(tmp_path / "lc"))
    assert store.similarity_search("alpha", k=1)[0].page_content == "alpha"


# ---- LlamaIndex -----------------------------------------------------

def test_llamaindex_add_query_delete(db, name):
    pytest.importorskip("llama_index.core")
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
    from llama_index.core.vector_stores.types import (
        MetadataFilter, MetadataFilters, VectorStoreQuery)
    from tidevec.integrations.llamaindex import TideVecVectorStore

    emb = HashingEmbedder(128)

    def node(text, doc, **meta):
        n = TextNode(text=text, metadata=meta, embedding=emb.embed_query(text))
        n.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc)
        return n

    store = TideVecVectorStore(db, name)
    nodes = [node("cats purr when happy", "doc-cats", animal="cat"),
             node("dogs bark at strangers", "doc-dogs", animal="dog"),
             node("cats sleep sixteen hours", "doc-cats", animal="cat")]
    assert store.add(nodes) == [n.node_id for n in nodes]

    res = store.query(VectorStoreQuery(query_embedding=emb.embed_query("cats purr"),
                                       similarity_top_k=2))
    assert res.nodes[0].get_content() == "cats purr when happy"
    assert res.nodes[0].ref_doc_id == "doc-cats"
    assert len(res.similarities) == len(res.ids) == 2

    only_dogs = store.query(VectorStoreQuery(
        query_embedding=emb.embed_query("cats"), similarity_top_k=3,
        filters=MetadataFilters(filters=[MetadataFilter(key="animal", value="dog")])))
    assert [n.get_content() for n in only_dogs.nodes] == ["dogs bark at strangers"]

    store.delete("doc-cats")
    left = store.query(VectorStoreQuery(query_embedding=emb.embed_query("cats"),
                                        similarity_top_k=5))
    assert [n.ref_doc_id for n in left.nodes] == ["doc-dogs"]
