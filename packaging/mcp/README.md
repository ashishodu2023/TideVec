# MCP directory listings

Metadata for listing `tidevec-mcp` in MCP directories. Nothing here has
been published.

| File | For |
|---|---|
| `server.json` | [Official MCP Registry](https://registry.modelcontextprotocol.io) (`mcp-publisher publish`) |
| `smithery.yaml` + `Dockerfile` | [Smithery](https://smithery.ai) (container runtime, stdio) |
| `../pypi/tidevec-mcp/` | The thin PyPI package that `uvx tidevec-mcp` installs |

Client setup docs: [`examples/mcp/README.md`](../../examples/mcp/README.md).

## Release steps

1. Publish `tidevec` and `tidevec-embedded` X.Y.Z, then the launcher:
   `python -m build packaging/pypi/tidevec-mcp && twine upload dist/tidevec_mcp-*`.
   Its README carries the `mcp-name: io.github.ashishodu2023/tidevec`
   marker the registry uses to verify PyPI ownership.
2. Bump `version` (twice) in `server.json`, then
   `mcp-publisher login github && mcp-publisher publish` from this directory.
3. Smithery: connect the GitHub repo and set the base directory to
   `packaging/mcp`.
4. mcp.so / Glama / PulseMCP: submit the repository URL. They read the
   README and `server.json`.

Bump the version in `../pypi/tidevec-mcp/pyproject.toml` (both `version`
and the `tidevec[...]==` pin), `server.json` and `Dockerfile` together.
