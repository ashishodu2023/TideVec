"""
LangChain retrieval that doesn't quote last quarter's policy.

    pip install "tidevec[embedded,langchain]"
    python examples/02_langchain_rag.py            # offline, hashing embeddings
    OPENAI_API_KEY=... python examples/02_langchain_rag.py --openai

Builds a small knowledge base where two policies were revised, then
compares the context a RAG chain would receive from a plain similarity
retriever against TideVec's (recency-aware, superseded versions removed).
Plug the retriever into any LangChain chain via ``store.as_retriever()``.
"""

import argparse
import shutil
import tempfile
import time

from tidevec import Local
from tidevec.integrations.langchain import TideVecVectorStore

DAY = 86_400_000
now = int(time.time() * 1000)

KB = [
    # (text, doc_id, age in days)
    ("Remote work: employees may work remotely up to 2 days per week.", "remote-work", 300),
    ("Remote work: employees may now work remotely up to 4 days per week.", "remote-work", 10),
    ("Expenses: meals are reimbursed up to 25 USD per day while travelling.", "expenses", 400),
    ("Expenses: the meal allowance while travelling is now 40 USD per day.", "expenses", 30),
    ("Security: laptops must use full-disk encryption.", "security", 200),
    ("Holidays: the office is closed between Christmas and New Year.", "holidays", 90),
]

parser = argparse.ArgumentParser()
parser.add_argument("--openai", action="store_true", help="use OpenAI embeddings")
args = parser.parse_args()

if args.openai:
    from tidevec.embeddings import OpenAIEmbedder
    embeddings = OpenAIEmbedder()
    threshold = 0.85
else:
    from tidevec.embeddings import HashingEmbedder
    embeddings = HashingEmbedder(384)
    threshold = 0.5

path = tempfile.mkdtemp(prefix="tidevec-langchain-")
store = TideVecVectorStore(
    Local(path), "handbook", embeddings,
    half_life_ms=90 * DAY,
    supersession=True, supersede_threshold=threshold, supersede_match_field="doc_id",
)
for text, doc_id, age in KB:
    store.add_texts([text], metadatas=[{"doc_id": doc_id}], created_at=now - age * DAY)

questions = [
    "How many days per week can employees work remotely?",
    "What is the meal allowance per day when travelling?",
]
for q in questions:
    print(f"\nQ: {q}")
    plain = store.similarity_search(q, k=2, temporal_blend=0.0)
    print("  plain similarity context:")
    for d in plain:
        print(f"    - {d.page_content}")
    current = store.similarity_search(q, k=2, exclude_superseded=True)
    print("  TideVec context:")
    for d in current:
        print(f"    - {d.page_content}")

retriever = store.as_retriever(search_kwargs={"k": 3, "exclude_superseded": True})
print("\nretriever:", [d.metadata["doc_id"] for d in retriever.invoke("remote work policy")])

shutil.rmtree(path)
