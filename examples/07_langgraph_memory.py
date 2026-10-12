"""
LangGraph long-term memory on TideVec: a BaseStore plus versioned memory tools.

    pip install "tidevec[embedded,langgraph]"
    python examples/07_langgraph_memory.py                    # embedded, offline
    python examples/07_langgraph_memory.py --url localhost:6399

Part 1 uses ``TideVecStore`` the way LangGraph agents use any store
(``create_react_agent(model, tools, store=store)``, or ``store`` injected
into graph nodes): namespaced items, semantic search, filters.

Part 2 gives an agent ``remember`` / ``recall`` / ``memory_history`` tools
backed by ``AgentMemory``. Remembering a fact under an existing key
supersedes the old version instead of leaving both. No LLM is needed
here; the tools are called directly.
"""

import argparse
import shutil
import tempfile
import time

from tidevec import Local, TideVec
from tidevec.embeddings import HashingEmbedder
from tidevec.integrations.langgraph import TideVecStore, memory_tools
from tidevec.memory import AgentMemory

parser = argparse.ArgumentParser()
parser.add_argument("--url", help="tidevec-server host:port (default: embedded)")
args = parser.parse_args()

path = None
if args.url:
    db = TideVec(args.url)
else:
    path = tempfile.mkdtemp(prefix="tidevec-langgraph-")
    db = Local(path)

suffix = int(time.time())
emb = HashingEmbedder(256)

# ---- Part 1: BaseStore ----------------------------------------------------
store = TideVecStore(db, collection=f"lg_store_{suffix}",
                     index={"dims": 256, "embed": emb.embed_documents, "fields": ["text"]})

user = ("users", "alice", "memories")
store.put(user, "diet", {"text": "Alice is vegetarian", "kind": "preference"})
store.put(user, "city", {"text": "Alice lives in Porto", "kind": "fact"})
store.put(("users", "bob", "memories"), "diet", {"text": "Bob is allergic to nuts", "kind": "preference"})

print("search(users/alice, 'what should I cook for her'):")
for item in store.search(("users", "alice"), query="what food should I cook for her", limit=2):
    print(f"  {item.score:.3f}  {item.key}: {item.value['text']}")

store.put(user, "diet", {"text": "Alice is vegan now", "kind": "preference"})
item = store.get(user, "diet")
print(f"\nget(diet) -> {item.value['text']!r} "
      f"(created {item.created_at:%H:%M:%S}, updated {item.updated_at:%H:%M:%S})")
print("preferences across users:",
      [(i.namespace[1], i.value["text"]) for i in store.search(("users",), filter={"kind": "preference"})])
print("namespaces:", store.list_namespaces(prefix=("users",)))

# ---- Part 2: versioned memory tools ----------------------------------------
memory = AgentMemory(db, f"lg_memory_{suffix}", embedder=emb)
remember, recall, history = memory_tools(memory, namespace_key="alice")

print("\n" + remember.invoke({"text": "Alice's manager is Dana", "key": "manager"}))
print(remember.invoke({"text": "Alice's manager is Priya since the reorg", "key": "manager"}))
print("recall('who is her manager'):\n" + recall.invoke({"query": "who is Alice's manager"}))
print("memory_history('manager'):\n" + history.invoke({"key": "manager"}))

if path:
    db.close()
    shutil.rmtree(path)
else:
    db.drop_collection(f"lg_store_{suffix}")
    db.drop_collection(f"lg_memory_{suffix}")
