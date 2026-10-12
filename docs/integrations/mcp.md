# Agent memory and MCP

## `AgentMemory`

`tidevec.memory.AgentMemory` is a small long-term memory API on top of a TideVec collection. A `key`
names the fact being remembered; a new memory with the same key supersedes the previous one, so
`recall` returns the current value and `history` keeps the old ones.

```python
from tidevec import Local
from tidevec.memory import AgentMemory

mem = AgentMemory(Local("~/.tidevec/memory"))
mem.remember("User's favourite editor is Vim", key="user.editor")
mem.remember("User switched to Helix", key="user.editor")   # supersedes the Vim memory
mem.recall("which editor does the user like?")              # -> the Helix memory only
mem.history("user.editor")                                  # both, oldest first
mem.forget(key="user.editor")                               # delete every version
```

`AgentMemory(db, collection="agent_memory", embedder=None, *, half_life_ms=HalfLife.ONE_MONTH,
temporal_blend=0.2, supersede_threshold=0.95, contradiction_judge=None)`

- `db` is `TideVec` or `Local`. `embedder` defaults to `tidevec.embeddings.default_embedder()`
  (sentence-transformers if installed, else OpenAI if `OPENAI_API_KEY` is set, else a lexical hashing
  embedder with a warning).
- Keyed memories are linked explicitly with `UPDATES` edges. Memories without a key are linked by
  TideVec's similarity-based [supersession](../concepts/supersession.md) when they are near-identical
  restatements (`supersede_threshold`, `vector_score` 0.95 = cosine 0.9).
- `remember(text, key=None, metadata=None, created_at=None)` returns `{"id", "superseded"}`, plus
  `"relations"` when a `contradiction_judge` is set (see
  [contradictions](../concepts/edges.md#contradictions)).
- `recall(query, k=5, *, include_superseded=False, key=None)` returns dicts with `id`, `text`, `key`,
  `score`, `created_at`, `age`, `stale`, `superseded_by`, `contradicted_by` and `metadata`.

## `tidevec-mcp`

`tidevec-mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server that exposes
`AgentMemory` to MCP clients such as Claude Desktop and Cursor.

```bash
pip install "tidevec[mcp,embedded]"
pip install sentence-transformers   # optional, recommended: real local embeddings
```

| Tool | What it does |
|---|---|
| `remember(text, key?, metadata?)` | Store a memory; the same key supersedes the old value. |
| `recall(query, k?, include_superseded?)` | Most relevant current memories, ranked by similarity blended with recency. |
| `history(key)` | Every version of a fact, oldest first. |
| `forget(memory_id? \| key?)` | Delete a memory or every version of a key. |

The server's instructions tell the model to `recall` before answering questions that may depend on
earlier conversations, and to pass a stable `key` (`user.timezone`, `project.db_choice`) when it
remembers something.

### Claude Desktop

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

There is no embedded wheel for Windows yet. On Windows, run `tidevec-server` in Docker or WSL2 and
use `"args": ["--url", "localhost:6399"]`, with `TIDEVEC_API_KEY` in `"env"` if auth is on.

### Cursor

`.cursor/mcp.json` in a project, or `~/.cursor/mcp.json` for all projects:

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

### Options

| Option | Default | |
|---|---|---|
| `--path DIR` | `~/.tidevec/memory` (or `$TIDEVEC_MEMORY_PATH`) | embedded database directory |
| `--url HOST:PORT` | `$TIDEVEC_URL` | use a running server instead of embedded mode |
| `--api-key KEY` | `$TIDEVEC_API_KEY` | server API key |
| `--collection NAME` | `agent_memory` | |
| `--embedder KIND` | `auto` | `auto`, `hashing`, `openai`, `sentence-transformers` |
| `--half-life-days N` | 30 | how fast old memories lose rank |
| `--transport T` | `stdio` | `stdio` or `streamable-http` |

Pointing several agents at the same `--url` gives them a shared memory. In embedded mode, only one
process may use a `--path` at a time.

### Try it without an MCP client

```bash
python examples/04_agent_memory/demo.py
```

The demo stores a project deadline and preferred language, updates both, and shows that `recall`
returns only the current values while `history` keeps the changes.
