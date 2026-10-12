# @tidevec/ai-sdk

[Vercel AI SDK](https://ai-sdk.dev) tools and RAG middleware for [TideVec](https://github.com/ashishodu2023/TideVec),
the vector database that knows when a fact was replaced.

- **Memory tools** (`remember`, `recall`, `history`, `forget`): facts stored under the same key
  supersede each other, so the agent recalls "the user switched to Helix" instead of both that and
  "the user likes Vim".
- **Knowledge search tool** and **RAG middleware** that drop document versions superseded by newer ones.

Works with AI SDK 5, 6 and 7 (`tool({ inputSchema })`) and zod 3.25+ or 4.

```bash
npm install @tidevec/ai-sdk tidevec ai zod
docker run -p 6399:6399 averm004/tidevec:0.3.0
```

## Agent memory

```ts
import { generateText, stepCountIs } from "ai";
import { openai } from "@ai-sdk/openai";
import { tidevecMemoryTools, MEMORY_INSTRUCTIONS } from "@tidevec/ai-sdk";

const tools = tidevecMemoryTools({
  client: { url: "localhost:6399", apiKey: process.env.TIDEVEC_API_KEY },
  embedding: openai.embedding("text-embedding-3-small"),
  collection: "agent_memory",
});

const { text } = await generateText({
  model: openai("gpt-4o-mini"),
  system: MEMORY_INSTRUCTIONS,
  tools,
  stopWhen: stepCountIs(5),
  prompt: "I moved from Vim to Helix last week. What editor do I use?",
});
```

Memories use the same layout as the Python `AgentMemory` and the `tidevec-mcp` server, so an agent built
with this package and Claude Desktop/Cursor (through MCP) can share one memory collection, provided they
use the same embedding model.

## Retrieval

```ts
import { wrapLanguageModel } from "ai";
import { addDocuments, tidevecRagMiddleware, tidevecSearchTool } from "@tidevec/ai-sdk";

const kb = { collection: "kb", embedding: openai.embedding("text-embedding-3-small") };

// Index documents; a new text with the same doc_id supersedes the old one.
await addDocuments({ ...kb, createIfMissing: { dim: 1536, matchField: "doc_id" } }, [
  { id: "refunds-2026-10", text: "Refunds take 3 days.", metadata: { doc_id: "refunds" } },
]);

// Option 1: let the model decide when to search.
const tools = { searchKnowledge: tidevecSearchTool(kb) };

// Option 2: always inject the current context as a system message.
const model = wrapLanguageModel({ model: openai("gpt-4o-mini"), middleware: tidevecRagMiddleware({ ...kb, topK: 4 }) });
```

`createRetriever(options)` returns the underlying `(query) => RetrievedDocument[]` function.

## Options

| Option | Default | Meaning |
|---|---|---|
| `client` | `http://localhost:6399` | `TideVec` instance, options object or URL |
| `embedding` | required | AI SDK embedding model (or gateway id), a `tidevec` `Embedder`, or `(text) => number[]` |
| `collection` | `agent_memory` for memory | collection name |
| `textKey` | `text` | payload key holding the document text |
| `topK` | 5 | documents per query |
| `excludeSuperseded` | `true` | drop documents a newer version supersedes |
| `temporalBlend` | collection setting | recency weight per query |
| `filter` | none | fixed payload filter, e.g. `{ tenant: "acme" }` |
| `formatContext` | dated list | middleware only: build the system message |

## Module formats

ESM and CommonJS builds are included. AI SDK 7 is ESM-only, so the CommonJS build needs Node 20.19+ or
22+ (which can `require()` ES modules) when used with it.

## Development

```bash
cd sdk/typescript && npm install && npm run build --workspace tidevec
npm test --workspace @tidevec/ai-sdk   # spawns build*/tidevec-server or $TIDEVEC_SERVER_BIN
```
