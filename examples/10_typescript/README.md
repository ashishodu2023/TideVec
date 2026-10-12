# TypeScript examples

| File | Shows | Needs |
|---|---|---|
| `quickstart.mjs` | Collections with supersession, search with superseded versions dropped, structured filters, `AgentMemory` | a TideVec server |
| `ai_sdk_agent.mjs` | Vercel AI SDK agent with `@tidevec/ai-sdk` memory tools and a knowledge-search tool | a server and `OPENAI_API_KEY` |

```bash
docker run -d -p 6399:6399 -e TIDEVEC_API_KEY=dev averm004/tidevec:0.3.0
cd examples/10_typescript && npm install
TIDEVEC_API_KEY=dev node quickstart.mjs
OPENAI_API_KEY=... TIDEVEC_API_KEY=dev node ai_sdk_agent.mjs
```

Until the packages are on npm, link the local builds instead of `npm install`:

```bash
(cd ../../sdk/typescript && npm install && npm run build)
npm install ../../sdk/typescript/packages/tidevec ../../sdk/typescript/packages/ai-sdk ai @ai-sdk/openai zod
```
