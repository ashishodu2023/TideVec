"""
TideVec agent-memory MCP server.

Gives any MCP client (Claude Desktop, Cursor, ...) a long-term memory that
knows when a fact has been updated:

    remember(text, key?)        store a memory; same key supersedes older versions
    recall(query, k?, filter?, as_of?)
                                current memories, ranked by relevance + recency;
                                as_of recalls what was known at that time
    history(key, as_of?)        every version of a key, oldest first
    forget(memory_id? | key?)   delete

Run (stdio):    tidevec-mcp
Options:        --path DIR          embedded database (default ~/.tidevec/memory)
                --url HOST:PORT     use a tidevec-server instead ($TIDEVEC_URL)
                --embedder auto|hashing|openai|sentence-transformers
                --transport stdio|streamable-http

Requires: pip install "tidevec[mcp,embedded]"
"""

from __future__ import annotations

import argparse
import os as _os
from typing import Any, Dict, List, Optional

from .memory import AgentMemory, default_memory_path

_INSTRUCTIONS = (
    "Long-term memory. Call `recall` before answering questions that may depend on "
    "earlier conversations. Call `remember` for durable facts about the user, project "
    "or decisions; pass a stable `key` (e.g. 'user.timezone', 'project.db_choice') so "
    "that a later update supersedes the old value instead of contradicting it."
)


def _server_class():
    try:
        from mcp.server.mcpserver import MCPServer  # mcp >= 2
        return MCPServer
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP  # mcp 1.x
        return FastMCP
    except ImportError as e:
        raise ImportError('pip install "tidevec[mcp]"') from e


def build_server(memory: AgentMemory, name: str = "tidevec-memory"):
    """Create the MCP server exposing ``memory``'s tools."""
    server = _server_class()(name, instructions=_INSTRUCTIONS)

    @server.tool()
    def remember(text: str, key: str = "",
                 metadata: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Store a memory. Use `key` to name the fact (e.g. 'user.editor'); a new
        memory with an existing key supersedes the previous value."""
        return memory.remember(text, key=key or None, metadata=metadata)

    @server.tool()
    def recall(query: str, k: int = 5, include_superseded: bool = False,
               key: str = "", filter: Optional[Dict[str, Any]] = None,
               as_of: str = "") -> List[Dict[str, Any]]:
        """Return the memories most relevant to `query`. Outdated versions of a
        fact are hidden unless include_superseded is true.

        filter: match metadata, e.g. {"project": "alpha"} or
            {"priority": {"$gte": 2}}; operators $eq $ne $gt $gte $lt $lte
            $in $nin $exists, combinable with $and / $or / $not.
        as_of: ISO 8601 date or time (or epoch ms) to recall what was known
            then; a bare date means the end of that day (UTC)."""
        return memory.recall(query, k, include_superseded=include_superseded,
                             key=key or None, filter=filter or None, as_of=as_of or None)

    @server.tool()
    def history(key: str, as_of: str = "") -> List[Dict[str, Any]]:
        """Every stored version of the fact named `key`, oldest first. With
        `as_of` (ISO 8601 or epoch ms), only versions known at that time."""
        return memory.history(key, as_of=as_of or None)

    @server.tool()
    def forget(memory_id: str = "", key: str = "") -> Dict[str, int]:
        """Delete one memory by id, or every version of a key."""
        return {"deleted": memory.forget(memory_id or None, key or None)}

    return server


def _make_embedder(kind: str):
    from . import embeddings
    if kind == "hashing":
        return embeddings.HashingEmbedder()
    if kind == "openai":
        return embeddings.OpenAIEmbedder()
    if kind == "sentence-transformers":
        return embeddings.SentenceTransformerEmbedder()
    return embeddings.default_embedder()


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="tidevec-mcp", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", default=default_memory_path())
    p.add_argument("--url", default=_os.environ.get("TIDEVEC_URL", ""))
    p.add_argument("--api-key", default=_os.environ.get("TIDEVEC_API_KEY", ""))
    p.add_argument("--collection", default="agent_memory")
    p.add_argument("--embedder", default="auto",
                   choices=["auto", "hashing", "openai", "sentence-transformers"])
    p.add_argument("--half-life-days", type=float, default=30.0)
    p.add_argument("--transport", default="stdio", choices=["stdio", "streamable-http"])
    args = p.parse_args(argv)

    if args.url:
        from .client import TideVec
        db: Any = TideVec(args.url, api_key=args.api_key)
    else:
        from .local import Local
        db = Local(_os.path.expanduser(args.path))

    memory = AgentMemory(db, args.collection, _make_embedder(args.embedder),
                         half_life_ms=int(args.half_life_days * 86_400_000))
    build_server(memory).run(transport=args.transport)


if __name__ == "__main__":
    main()
