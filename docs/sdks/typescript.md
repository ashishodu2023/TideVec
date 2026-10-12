# TypeScript SDK

[`sdk/typescript`](https://github.com/ashishodu2023/TideVec/tree/main/sdk/typescript) is an npm
workspace with three packages, built and tested in CI on Node 20 and 22 (unit tests plus live-server
tests):

| Package | What it is |
|---|---|
| [`tidevec`](https://github.com/ashishodu2023/TideVec/tree/main/sdk/typescript/packages/tidevec) | REST client, `AgentMemory`, `HashingEmbedder`. Fetch-based, no dependencies, ESM + CommonJS; Node 18+, Deno, Bun, edge runtimes, browsers |
| [`@tidevec/ai-sdk`](https://github.com/ashishodu2023/TideVec/tree/main/sdk/typescript/packages/ai-sdk) | Vercel AI SDK memory tools (`remember` / `recall` / `history` / `forget`), a knowledge-search tool and RAG middleware that drops superseded versions |
| [`n8n-nodes-tidevec`](https://github.com/ashishodu2023/TideVec/tree/main/sdk/typescript/packages/n8n-nodes-tidevec) | n8n community node (upsert and search; usable as an AI Agent tool). Early: not a LangChain vector-store sub-node yet |

!!! warning "Not on npm yet"
    The packages are versioned 0.3.0 but have not been published. Until they are, build them from a
    checkout and install the local paths.

```bash
cd sdk/typescript
npm ci && npm run build
npm install /path/to/TideVec/sdk/typescript/packages/tidevec      # from your project
```

```ts
import { TideVec, HalfLife } from "tidevec";

const db = new TideVec({ url: "localhost:6399", apiKey: process.env.TIDEVEC_API_KEY });
await db.createCollection("kb", {
  dim: 384,
  temporal: { halfLifeMs: HalfLife.ONE_MONTH, temporalBlend: 0.3 },
  supersession: { matchField: "doc_id", similarityThreshold: 0.9 },
});
const { hits } = await db.search("kb", queryVector, {
  topK: 5,
  excludeSuperseded: true,
  filter: { published: { $gte: "2025-01-01" } },
  asOf: "2026-01-01T00:00:00Z",
});
```

The client covers the same REST surface as the Python one: `get`, `getMany`, `scrollPage`, `listIds`,
`count`, aliases, keyword and hybrid search (`text`, `hybrid`), DriftBridge and backups. See
[Retrieval, filters, as-of and hybrid](../concepts/filters.md) and the package READMEs for the full
API. Runnable examples: [`examples/10_typescript/`](https://github.com/ashishodu2023/TideVec/tree/main/examples/10_typescript).
