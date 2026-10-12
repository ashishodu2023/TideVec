"""
CrewAI unified Memory stored in TideVec.

    pip install "tidevec[embedded,crewai]"
    python examples/08_crewai_memory.py                       # embedded, offline
    python examples/08_crewai_memory.py --url localhost:6399

``TideVecStorage`` implements CrewAI's ``StorageBackend``. Pass it as
``Memory(storage=...)`` (then ``Crew(memory=memory)``), or call
``storage.install_as_default()`` so every ``Memory()`` uses it. This
example saves pre-classified records and runs shallow recall, which needs
no LLM. ``memory.remember(...)`` and deep recall use your crew's LLM to
classify and to plan queries.
"""

import argparse
import os
import shutil
import tempfile
import time
from datetime import datetime, timedelta

os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import Memory  # noqa: E402
from crewai.memory.types import MemoryRecord  # noqa: E402

from tidevec import Local, TideVec  # noqa: E402
from tidevec.embeddings import HashingEmbedder  # noqa: E402
from tidevec.integrations.crewai import TideVecStorage  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--url", help="tidevec-server host:port (default: embedded)")
args = parser.parse_args()

path = None
if args.url:
    db = TideVec(args.url)
else:
    path = tempfile.mkdtemp(prefix="tidevec-crewai-")
    db = Local(path)

emb = HashingEmbedder(256)
storage = TideVecStorage(db, collection=f"crew_memory_{int(time.time())}")
memory = Memory(storage=storage, embedder=emb.embed_documents, llm="openai/gpt-4o-mini")


def record(text, scope, categories, days_ago=0, importance=0.5):
    return MemoryRecord(content=text, scope=scope, categories=categories, importance=importance,
                        embedding=emb.embed_query(text),
                        created_at=datetime.utcnow() - timedelta(days=days_ago))


storage.save([
    record("Launch of project Atlas is planned for March 3", "/projects/atlas", ["planning"], 40),
    record("Atlas launch moved to April 14 after the security review", "/projects/atlas",
           ["planning"], 2, importance=0.9),
    record("Atlas uses Postgres 16 with logical replication", "/projects/atlas/tech", ["architecture"], 20),
    record("Team offsite is in Lisbon in May", "/company", ["events"], 10),
])

print("scopes under /projects:", storage.list_scopes("/projects"))
info = storage.get_scope_info("/projects/atlas")
print(f"/projects/atlas: {info.record_count} records, categories {info.categories}")

print("\nrecall('when does Atlas launch', depth='shallow'):")
for m in memory.recall("when does Atlas launch", depth="shallow", limit=2):
    print(f"  {m.score:.3f}  {m.record.content}  [{m.record.scope}]")

print("\nvector search restricted to the planning category:")
for rec, score in storage.search(emb.embed_query("Atlas launch date"), categories=["planning"], limit=3):
    print(f"  {score:.3f}  {rec.content}")

storage.reset()
if path:
    db.close()
    shutil.rmtree(path)
