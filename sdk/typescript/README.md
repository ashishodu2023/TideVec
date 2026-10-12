# TideVec JavaScript / TypeScript packages

npm workspaces for the TideVec JS ecosystem:

| Package | Path | What it is |
|---|---|---|
| [`tidevec`](packages/tidevec) | `packages/tidevec` | REST client, `AgentMemory`, `HashingEmbedder` (no dependencies, ESM + CJS, Node 18+/browsers) |
| [`@tidevec/ai-sdk`](packages/ai-sdk) | `packages/ai-sdk` | Vercel AI SDK memory tools, search tool and RAG middleware |
| [`n8n-nodes-tidevec`](packages/n8n-nodes-tidevec) | `packages/n8n-nodes-tidevec` | n8n community node (skeleton) |

```bash
cd sdk/typescript
npm install
npm run build        # tidevec first; the others import its dist/
npm run typecheck
npm test             # integration tests spawn a tidevec-server (see below)
```

Integration tests look for the server binary in `$TIDEVEC_SERVER_BIN`, `<repo>/build-eco/`, `<repo>/build/`
and `$PATH`, and are skipped when none is found. Set `TIDEVEC_REQUIRE_SERVER=1` in CI to make a missing
binary an error.

Release order: `tidevec`, then `@tidevec/ai-sdk` and `n8n-nodes-tidevec` (`npm publish --workspace ...`
with provenance). Nothing here has been published yet.
