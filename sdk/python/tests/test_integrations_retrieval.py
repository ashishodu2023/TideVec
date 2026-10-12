"""LangChain / LlamaIndex: get-by-id and rich filters."""

import pytest

from tidevec.embeddings import HashingEmbedder


def test_langchain_get_by_ids_and_rich_filters(db, name):
    pytest.importorskip("langchain_core")
    from tidevec.integrations.langchain import TideVecVectorStore

    store = TideVecVectorStore(db, name, HashingEmbedder(128))
    ids = store.add_texts([f"note {i}" for i in range(6)],
                          metadatas=[{"page": i, "lang": "en" if i % 2 else "de",
                                      "tags": ["x"]} for i in range(6)])

    docs = store.get_by_ids([ids[3], "missing", ids[0]])
    assert [d.id for d in docs] == [ids[3], ids[0]]
    assert docs[0].page_content == "note 3"
    assert docs[0].metadata["page"] == 3 and docs[0].metadata["tags"] == ["x"]
    assert store.get_by_ids([]) == []

    hits = store.similarity_search("note", k=10, filter={"page": {"$gte": 2}, "lang": "en"})
    assert sorted(d.metadata["page"] for d in hits) == [3, 5]
    hits = store.similarity_search("note", k=10,
                                   filter={"$or": [{"page": 0}, {"page": {"$in": [4, 5]}}]})
    assert sorted(d.metadata["page"] for d in hits) == [0, 4, 5]
    assert len(store.similarity_search("note", k=10, filter="lang=de")) == 3

    assert store.delete(filter={"page": {"$lt": 2}})
    assert sorted(d.metadata["page"] for d in store.similarity_search("note", k=10)) == [2, 3, 4, 5]


def test_llamaindex_filters_get_nodes_and_delete_nodes(db, name):
    pytest.importorskip("llama_index.core")
    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode
    from llama_index.core.vector_stores.types import (
        FilterCondition, FilterOperator, MetadataFilter, MetadataFilters, VectorStoreQuery)
    from tidevec.integrations.llamaindex import TideVecVectorStore

    emb = HashingEmbedder(128)

    def node(text, doc, **meta):
        n = TextNode(text=text, metadata=meta, embedding=emb.embed_query(text))
        n.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=doc)
        return n

    store = TideVecVectorStore(db, name)
    nodes = [node(f"chunk {i}", f"doc-{i % 3}", year=2020 + i, kind="a" if i < 3 else "b")
             for i in range(6)]
    store.add(nodes)
    q = emb.embed_query("chunk")

    def query(**kw):
        res = store.query(VectorStoreQuery(query_embedding=q, similarity_top_k=10, **kw))
        return sorted(n.metadata["year"] for n in res.nodes)

    F, Fs = MetadataFilter, MetadataFilters
    assert query(filters=Fs(filters=[F(key="year", value=2022, operator=FilterOperator.GTE),
                                     F(key="kind", value="b", operator=FilterOperator.NE)])) \
        == [2022]
    assert query(filters=Fs(condition=FilterCondition.OR, filters=[
        F(key="year", value=2020),
        Fs(filters=[F(key="kind", value=["b"], operator=FilterOperator.IN),
                    F(key="year", value=2024, operator=FilterOperator.LT)])])) == [2020, 2023]
    assert query(filters=Fs(condition=FilterCondition.NOT,
                            filters=[F(key="kind", value="a")])) == [2023, 2024, 2025]
    assert query(filters=Fs(filters=[F(key="missing", value=None,
                                       operator=FilterOperator.IS_EMPTY)])) == list(range(2020, 2026))
    assert query(node_ids=[nodes[1].node_id, nodes[4].node_id]) == [2021, 2024]
    assert query(doc_ids=["doc-0"]) == [2020, 2023]
    with pytest.raises(NotImplementedError):
        query(filters=Fs(filters=[F(key="kind", value="a", operator=FilterOperator.TEXT_MATCH)]))

    got = store.get_nodes(node_ids=[nodes[2].node_id, nodes[0].node_id])
    assert [n.get_content() for n in got] == ["chunk 2", "chunk 0"]
    got = store.get_nodes(filters=Fs(filters=[F(key="kind", value="b")]))
    assert sorted(n.get_content() for n in got) == ["chunk 3", "chunk 4", "chunk 5"]

    store.delete_nodes(filters=Fs(filters=[F(key="year", value=2024,
                                             operator=FilterOperator.GTE)]))
    store.delete_nodes(node_ids=[nodes[0].node_id])
    assert query() == [2021, 2022, 2023]
