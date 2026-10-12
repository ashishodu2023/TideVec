# tidevec-mcp

<!-- mcp-name: io.github.ashishodu2023/tidevec -->

Long-term memory for MCP clients (Claude Desktop, Claude Code, Cursor,
Windsurf, Zed, VS Code) that knows when a fact has changed. Remembering
`user.editor = "Neovim"` after `user.editor = "VS Code"` supersedes the
old value instead of leaving two contradicting memories.

```bash
uvx tidevec-mcp                      # embedded database in ~/.tidevec/memory
uvx tidevec-mcp --url localhost:6399 # or a tidevec-server
```

Tools: `remember(text, key?)`, `recall(query, k?)`, `history(key)`,
`forget(memory_id? | key?)`.

Options: `--path DIR`, `--url HOST:PORT` (`$TIDEVEC_URL`), `--api-key`
(`$TIDEVEC_API_KEY`), `--collection`, `--embedder auto|hashing|openai|sentence-transformers`,
`--half-life-days`, `--transport stdio|streamable-http`.

For semantic (not just lexical) recall, install an embedder:
`uvx --from "tidevec-mcp[local-embeddings]" tidevec-mcp`, or set
`OPENAI_API_KEY` (and optionally `--embedder openai`).

Client configuration: https://github.com/ashishodu2023/TideVec/tree/main/examples/mcp

This package only provides the `tidevec-mcp` command; the implementation
is `tidevec.mcp_server` in the [`tidevec`](https://pypi.org/project/tidevec/) package.
