"""
Agent memory that knows when a fact changed.

    pip install "tidevec[embedded]"
    python examples/04_agent_memory/demo.py

The same AgentMemory class powers the `tidevec-mcp` server (see README.md
in this folder for Claude Desktop / Cursor configuration).
"""

import shutil
import tempfile

from tidevec import Local
from tidevec.embeddings import HashingEmbedder
from tidevec.memory import AgentMemory

path = tempfile.mkdtemp(prefix="tidevec-memory-")
mem = AgentMemory(Local(path), embedder=HashingEmbedder(256))

mem.remember("The user's preferred language for examples is Python", key="user.language")
mem.remember("The project deadline is March 3", key="project.deadline")
mem.remember("The team decided to use Postgres for the main database", key="project.database")

# Weeks later, things change.
mem.remember("The user now prefers TypeScript for examples", key="user.language")
mem.remember("The project deadline moved to April 9", key="project.deadline")

print("recall('what language should examples use?'):")
for m in mem.recall("what language should examples use?", k=2):
    print(f"  [{m['key']}] {m['text']}")

print("\nrecall('when is the deadline?'):")
for m in mem.recall("when is the deadline?", k=1):
    print(f"  [{m['key']}] {m['text']}")

print("\nhistory('project.deadline'):")
for m in mem.history("project.deadline"):
    status = "superseded" if m["superseded_by"] else "current"
    print(f"  {status:<10} {m['text']}")

shutil.rmtree(path)
