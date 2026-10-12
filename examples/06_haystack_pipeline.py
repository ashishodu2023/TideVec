"""
Haystack RAG pipeline whose retriever never returns an outdated document version.

    pip install "tidevec[embedded,haystack]"
    python examples/06_haystack_pipeline.py                   # embedded, offline
    python examples/06_haystack_pipeline.py --url localhost:6399

Documents carry a ``doc_id``. Writing a new version of a document makes
TideVec link it to the old one (supersession), and
``TideVecEmbeddingRetriever`` drops the superseded version. Filters use
Haystack's normal syntax.
"""

import argparse
import shutil
import tempfile
import time

from haystack import Document, Pipeline

from tidevec import Local, TideVec
from tidevec.embeddings import HashingEmbedder
from tidevec.integrations.haystack import TideVecDocumentStore, TideVecEmbeddingRetriever

DAY = 86_400_000
now = int(time.time() * 1000)

parser = argparse.ArgumentParser()
parser.add_argument("--url", help="tidevec-server host:port (default: embedded)")
args = parser.parse_args()

path = None
if args.url:
    db = TideVec(args.url)
else:
    path = tempfile.mkdtemp(prefix="tidevec-haystack-")
    db = Local(path)

emb = HashingEmbedder(384)
store = TideVecDocumentStore(db, collection=f"handbook_{now}", embedding_dim=384,
                             supersede_match_field="doc_id", supersede_threshold=0.5)


def doc(id, text, doc_id, team, age_days):
    return Document(id=id, content=text, embedding=emb.embed_query(text),
                    meta={"doc_id": doc_id, "team": team, "created_at": now - age_days * DAY})


store.write_documents([
    doc("oncall-v1", "On-call: the primary rotates every week on Monday.", "oncall", "sre", 200),
    doc("deploy", "Deploys are frozen on Fridays after 3pm.", "deploy", "sre", 120),
    doc("pto", "PTO requests need two weeks of notice.", "pto", "people", 60),
])
# The rotation changed.
store.write_documents([
    doc("oncall-v2", "On-call: the primary now rotates every two weeks on Wednesday.", "oncall", "sre", 3),
])

pipe = Pipeline()
pipe.add_component("retriever", TideVecEmbeddingRetriever(document_store=store, top_k=2))

question = "How often does the on-call primary rotate?"
result = pipe.run({"retriever": {"query_embedding": emb.embed_query(question)}})
print(f"Q: {question}")
for d in result["retriever"]["documents"]:
    print(f"  {d.score:.3f}  {d.content}")

print("\nWith every version (exclude_superseded=False):")
for d in store.embedding_retrieval(emb.embed_query(question), top_k=3, exclude_superseded=False):
    newer = d.meta["tidevec"]["superseded_by"]
    print(f"  {d.id:10} {f'superseded by {newer}' if newer else 'current'}")

sre = store.filter_documents({"field": "meta.team", "operator": "==", "value": "sre"})
print("\nfilter_documents(team == sre):", sorted(d.id for d in sre))
print("count_documents():", store.count_documents())

if path:
    db.close()
    shutil.rmtree(path)
else:
    db.drop_collection(store.collection)
