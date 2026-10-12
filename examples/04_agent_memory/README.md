# Agent memory over MCP

`tidevec-mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server that gives an
assistant long-term memory with four tools:

| Tool | What it does |
|------|--------------|
| `remember(text, key?, metadata?)` | Store a memory. A `key` names the fact (`user.timezone`, `project.db`); a new memory with the same key supersedes the old one. |
| `recall(query, k?, include_superseded?)` | Most relevant *current* memories, ranked by similarity blended with recency. |
| `history(key)` | Every version of a fact, oldest first. |
| `forget(memory_id? \| key?)` | Delete a memory or all versions of a key. |

Most memory layers append facts forever, so "the deadline is March 3" and "the deadline moved to
April 9" both come back and the model has to guess which is true. TideVec links the versions, so
`recall` returns only the current one, and `history` still shows how it changed.

## Install

```bash
pip install "tidevec[mcp,embedded]"
pip install sentence-transformers   # optional: real local embeddings (recommended)
```

Without `sentence-transformers` or `OPENAI_API_KEY`, the server falls back to a lexical hashing
embedder and prints a warning.

## Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or
`%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "tidevec-memory": {
      "command": "tidevec-mcp",
      "args": ["--path", "~/.tidevec/memory"]
    }
  }
}
```

Windows has no embedded wheel yet. Run `tidevec-server` in Docker or WSL2 and point the MCP server at
it with `"args": ["--url", "localhost:6399"]` (plus `TIDEVEC_API_KEY` in `"env"` if auth is on).

## Cursor

`.cursor/mcp.json` in a project (or `~/.cursor/mcp.json` globally):

```json
{
  "mcpServers": {
    "tidevec-memory": {
      "command": "tidevec-mcp",
      "args": ["--embedder", "sentence-transformers"]
    }
  }
}
```

## Options

```
--path DIR            embedded database directory (default ~/.tidevec/memory, or $TIDEVEC_MEMORY_PATH)
--url HOST:PORT       use a running tidevec-server instead of embedded mode ($TIDEVEC_URL)
--api-key KEY         server API key ($TIDEVEC_API_KEY)
--collection NAME     default agent_memory
--embedder KIND       auto | hashing | openai | sentence-transformers
--half-life-days N    how fast old memories lose rank (default 30)
--transport T         stdio (default) | streamable-http
```

Pointing several agents at the same `--url` gives them a shared memory.

## Try it without an MCP client

```bash
python examples/04_agent_memory/demo.py
```
