"""
TideVec as a LlamaIndex vector store.

    pip install "tidevec[embedded,llamaindex]"
    python examples/03_llamaindex.py

Uses an offline hashing embedder so it runs without API keys; swap in any
LlamaIndex embedding model (OpenAIEmbedding, HuggingFaceEmbedding, ...).
"""

import shutil
import tempfile
from typing import List

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding

from tidevec import Local
from tidevec.embeddings import HashingEmbedder
from tidevec.integrations.llamaindex import TideVecVectorStore

_hash = HashingEmbedder(256)


class HashingEmbedding(BaseEmbedding):
    def _get_query_embedding(self, query: str) -> List[float]:
        return _hash.embed_query(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return _hash.embed_query(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return _hash.embed_query(query)


Settings.embed_model = HashingEmbedding()
Settings.llm = None  # retrieval only; set an LLM to use index.as_query_engine()

path = tempfile.mkdtemp(prefix="tidevec-llamaindex-")
vector_store = TideVecVectorStore(Local(path), "release_notes",
                                  supersession=True, supersede_threshold=0.5,
                                  supersede_match_field="component",
                                  exclude_superseded=True)

docs = [
    Document(text="The API rate limit is 100 requests per minute.",
             metadata={"component": "api-limits"}),
    Document(text="Webhooks retry failed deliveries three times.",
             metadata={"component": "webhooks"}),
]
index = VectorStoreIndex.from_documents(
    docs, storage_context=StorageContext.from_defaults(vector_store=vector_store))

# A later release changes the limit: the old node is superseded automatically.
index.insert(Document(text="The API rate limit is now 500 requests per minute.",
                      metadata={"component": "api-limits"}))

for node in index.as_retriever(similarity_top_k=2).retrieve("What is the API rate limit?"):
    print(f"{node.score:.3f}  {node.get_content()}")

# Deleting a source document removes all of its nodes.
index.delete_ref_doc(docs[1].doc_id)
shutil.rmtree(path)
