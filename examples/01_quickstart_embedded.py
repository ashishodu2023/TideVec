"""
TideVec in 60 seconds — embedded mode, no server.

    pip install "tidevec[embedded]"
    python examples/01_quickstart_embedded.py

Shows the two things a plain vector database gets wrong:
  1. an old document that matches the query slightly better outranks the
     current one (fixed by temporal scoring inside the ANN search);
  2. nothing tells you the old document has been replaced (fixed by
     supersession: TideVec links versions automatically on upsert).
"""

import shutil
import tempfile
import time

from tidevec import HalfLife, Local
from tidevec.embeddings import HashingEmbedder

DAY = 86_400_000
now = int(time.time() * 1000)

emb = HashingEmbedder(256)
path = tempfile.mkdtemp(prefix="tidevec-quickstart-")
db = Local(path)

db.create_collection(
    "policies", dim=emb.dim,
    half_life_ms=HalfLife.ONE_MONTH, temporal_blend=0.3,
    # Link a new upsert to older vectors with the same doc_id that it
    # closely resembles.
    supersession=True, supersede_threshold=0.5, supersede_match_field="doc_id",
)

docs = [
    ("refunds-2025", "Refund policy: refunds are issued within 14 days of the return request.",
     "refunds", now - 200 * DAY),
    ("shipping-2025", "Shipping policy: standard shipping takes 5 business days.",
     "shipping", now - 150 * DAY),
    ("refunds-2026", "Refund policy update: refunds are now issued within 3 days.",
     "refunds", now - 2 * DAY),
]
for vid, text, doc_id, ts in docs:
    db.upsert("policies", [{"id": vid, "embedding": emb.embed_query(text), "created_at": ts,
                            "payload": {"text": text, "doc_id": doc_id}}])

# Users phrase questions like the documents they've seen, which often
# means like the *old* document.
query = emb.embed_query("How many days after the return request are refunds issued?")


def show(title, hits):
    print(f"\n{title}")
    for h in hits:
        flag = f"  <- superseded by {h.superseded_by}" if h.superseded_by else ""
        print(f"  {h.score:.3f}  (sim {h.vector_score:.3f}, recency {h.temporal_score:.2f})  "
              f"{h.id}{flag}")


show("Pure similarity (what a plain vector DB returns):",
     db.search("policies", query, top_k=3, temporal_blend=0.0))
show("TideVec default (similarity blended with recency during the search):",
     db.search("policies", query, top_k=3))
show("Push superseded versions to the bottom:",
     db.search("policies", query, top_k=3, superseded_penalty=1.0))

db.close()
shutil.rmtree(path)
