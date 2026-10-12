# TypeScript SDK

!!! warning "In progress, not published"
    A TypeScript client is being developed in
    [`sdk/typescript`](https://github.com/ashishodu2023/TideVec/tree/main/sdk/typescript) as npm
    workspaces: `tidevec` (REST client, `AgentMemory`, `HashingEmbedder`; Node 18+ and browsers),
    `@tidevec/ai-sdk` (Vercel AI SDK tools and RAG middleware) and `n8n-nodes-tidevec` (skeleton).
    **None of these packages is on npm yet**, and their API may change before the first release.
    <!-- TODO(docs): replace with a real page (install, examples, API table) once the packages ship. -->

To try it from a checkout:

```bash
cd sdk/typescript
npm install
npm run build
```

Until it ships, JavaScript applications can call the [REST API](../api/rest.md) directly; every
operation is a JSON request with an `X-Api-Key` header.
