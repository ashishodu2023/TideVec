"""
DSPy RAG module with a TideVec retriever that skips outdated passages.

    pip install "tidevec[embedded,dspy]"
    python examples/09_dspy_rag.py                            # retrieval only, offline
    OPENAI_API_KEY=... python examples/09_dspy_rag.py --lm openai/gpt-4o-mini
    python examples/09_dspy_rag.py --url localhost:6399

``TideVecRetriever`` is a ``dspy.Retrieve`` module. Call it with a query
(or a list of queries) and you get a ``Prediction`` with ``passages``,
``ids``, ``scores`` and ``metadata``. Passages that a newer version of the
same document superseded are left out.
"""

import argparse
import shutil
import tempfile
import time

import dspy

from tidevec import Local, TideVec
from tidevec.embeddings import HashingEmbedder
from tidevec.integrations.dspy import TideVecRetriever

DAY = 86_400_000
now = int(time.time() * 1000)

parser = argparse.ArgumentParser()
parser.add_argument("--url", help="tidevec-server host:port (default: embedded)")
parser.add_argument("--lm", help="DSPy LM id, e.g. openai/gpt-4o-mini (default: retrieval only)")
args = parser.parse_args()

path = None
if args.url:
    db = TideVec(args.url)
else:
    path = tempfile.mkdtemp(prefix="tidevec-dspy-")
    db = Local(path)

collection = f"faq_{now}"
retrieve = TideVecRetriever(db, collection, embedder=HashingEmbedder(384), k=2,
                            supersession=True, supersede_match_field="doc_id",
                            supersede_threshold=0.3)
retrieve.add(["Support hours are 9am to 5pm CET on weekdays.",
              "Refunds are issued to the original payment method."],
             metadatas=[{"doc_id": "hours"}, {"doc_id": "refunds"}], created_at=now - 90 * DAY)
retrieve.add(["Support hours are now 8am to 8pm CET, seven days a week."],
             metadatas=[{"doc_id": "hours"}])

question = "What are the support hours?"
pred = retrieve(question)
print(f"Q: {question}")
for text, score in zip(pred.passages, pred.scores):
    print(f"  {score:.3f}  {text}")


class RAG(dspy.Module):
    def __init__(self, retriever):
        super().__init__()
        self.retrieve = retriever
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.answer(context=context, question=question)


if args.lm:
    dspy.configure(lm=dspy.LM(args.lm))
    print("\nRAG answer:", RAG(retrieve)(question=question).answer)
else:
    print("\n(pass --lm openai/gpt-4o-mini to run the full RAG module)")

if path:
    db.close()
    shutil.rmtree(path)
else:
    db.drop_collection(collection)
