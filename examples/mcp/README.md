# TideVec memory for MCP clients

`tidevec-mcp` gives any MCP client a long-term memory that knows when a
fact changed. Remembering `user.editor = Neovim` after
`user.editor = VS Code` **supersedes** the old memory: `recall` returns
only the current value and `history` shows both.

| Tool | What it does |
|---|---|
| `remember(text, key?, metadata?)` | Store a memory. Reusing a `key` supersedes the previous value. |
| `recall(query, k?, include_superseded?, key?)` | Most relevant current memories, ranked by similarity and recency. |
| `history(key)` | Every version of a key, oldest first. |
| `forget(memory_id? \| key?)` | Delete one memory or every version of a key. |

All configs below run the server with [`uvx`](https://docs.astral.sh/uv/)
(install uv once: `curl -LsSf https://astral.sh/uv/install.sh | sh`).
`uvx tidevec-mcp` installs `tidevec[mcp,embedded]` into a cached
environment, so no other setup is needed.

**Pick a storage mode:**

- **Embedded (default)**: the database lives in `~/.tidevec/memory`.
  Use this for one client on one machine.
- **Server**: add `"--url", "localhost:6399"` and set `TIDEVEC_API_KEY`.
  Several clients (say Claude Desktop *and* Cursor) can then share one
  memory, because an embedded database can only be opened by one process
  at a time.

**Embeddings:** if sentence-transformers is installed, it is used.
Otherwise, if `OPENAI_API_KEY` is set, OpenAI embeddings are used.
Otherwise TideVec falls back to a lexical hashing embedder and prints a
warning. For semantic recall without an API key, use
`uvx --from "tidevec-mcp[local-embeddings]" tidevec-mcp`. The first run
downloads a model of about 90 MB.

> Before `tidevec-mcp` is on PyPI, use
> `uvx --from "tidevec[mcp,embedded]" tidevec-mcp` (same flags), or a
> local checkout: `uvx --from /path/to/TideVec/sdk/python[mcp,embedded] tidevec-mcp`.

---

## Claude Desktop

Edit `claude_desktop_config.json`:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "tidevec": {
      "command": "uvx",
      "args": ["tidevec-mcp"]
    }
  }
}
```

Server mode:

```json
{
  "mcpServers": {
    "tidevec": {
      "command": "uvx",
      "args": ["tidevec-mcp", "--url", "localhost:6399"],
      "env": { "TIDEVEC_API_KEY": "your-key" }
    }
  }
}
```

Restart Claude Desktop afterwards. If it can't find `uvx`, use its
absolute path (`which uvx`), because GUI apps don't read your shell's `PATH`.

## Claude Code

```bash
# All projects (user scope):
claude mcp add tidevec --scope user -- uvx tidevec-mcp

# Server mode:
claude mcp add tidevec --scope user -e TIDEVEC_API_KEY=your-key -- \
  uvx tidevec-mcp --url localhost:6399

claude mcp list
```

To share the config with your team, commit a project-scoped `.mcp.json`:

```json
{
  "mcpServers": {
    "tidevec": {
      "command": "uvx",
      "args": ["tidevec-mcp", "--path", ".tidevec/memory"]
    }
  }
}
```

## Cursor

Global: `~/.cursor/mcp.json`. Per project: `.cursor/mcp.json`.

```json
{
  "mcpServers": {
    "tidevec": {
      "command": "uvx",
      "args": ["tidevec-mcp"],
      "env": {}
    }
  }
}
```

Then check **Settings → MCP**: the server should be listed with 4 tools.

## Windsurf

Edit `~/.codeium/windsurf/mcp_config.json` (or use **Windsurf Settings →
Cascade → MCP Servers → View raw config**):

```json
{
  "mcpServers": {
    "tidevec": {
      "command": "uvx",
      "args": ["tidevec-mcp"]
    }
  }
}
```

## Zed

Add the server to `settings.json` (**zed: open settings**):

```json
{
  "context_servers": {
    "tidevec": {
      "source": "custom",
      "command": "uvx",
      "args": ["tidevec-mcp"],
      "env": {}
    }
  }
}
```

Older Zed releases nest the command instead:
`"tidevec": { "command": { "path": "uvx", "args": ["tidevec-mcp"] } }`.

## VS Code (GitHub Copilot agent mode)

Per workspace: `.vscode/mcp.json`. For your user profile, run
**MCP: Open User Configuration**. VS Code uses a `servers` key and can
prompt for secrets:

```json
{
  "inputs": [
    {
      "type": "promptString",
      "id": "tidevec-api-key",
      "description": "TideVec API key (leave empty for embedded mode)",
      "password": true
    }
  ],
  "servers": {
    "tidevec": {
      "type": "stdio",
      "command": "uvx",
      "args": ["tidevec-mcp"],
      "env": { "TIDEVEC_API_KEY": "${input:tidevec-api-key}" }
    }
  }
}
```

## Remote / HTTP clients

Clients that connect to a URL instead of launching a process can use
Streamable HTTP:

```bash
uvx tidevec-mcp --transport streamable-http --url localhost:6399
```

---

## Options

| Flag | Env | Default |
|---|---|---|
| `--path DIR` | `TIDEVEC_MEMORY_PATH` | `~/.tidevec/memory` (embedded mode) |
| `--url HOST:PORT` | `TIDEVEC_URL` | unset, which means embedded mode |
| `--api-key KEY` | `TIDEVEC_API_KEY` | none |
| `--collection NAME` | | `agent_memory` |
| `--embedder auto\|hashing\|openai\|sentence-transformers` | `OPENAI_API_KEY` for `openai` | `auto` |
| `--half-life-days N` | | `30`: how fast old memories lose rank |
| `--transport stdio\|streamable-http` | | `stdio` |

## Prompting tip

The server's instructions already ask the model to call `recall` before
answering and to use stable keys. For better results, add something like
this to your client's rules or system prompt:

> Use the tidevec memory tools. Call `recall` before answering questions
> that may depend on earlier sessions. When the user states a durable
> preference or decision, call `remember` with a dotted key such as
> `user.timezone` or `project.db`. Reusing the key replaces the old value.

## Directory listings

Listing metadata for the MCP Registry (`server.json`) and Smithery
(`smithery.yaml`, `Dockerfile`) lives in
[`packaging/mcp/`](../../packaging/mcp/).
