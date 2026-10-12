import { generateText, stepCountIs, wrapLanguageModel } from "ai";
import { MockEmbeddingModelV4, MockLanguageModelV4 } from "ai/test";
import { HashingEmbedder, TideVec } from "tidevec";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { type RunningServer, hasServer, startServer } from "../../tidevec/test/server.js";
import {
  addDocuments,
  asEmbedder,
  createRetriever,
  tidevecMemoryTools,
  tidevecRagMiddleware,
  tidevecSearchTool,
} from "../src/index.js";

const DAY = 86_400_000;
const hashing = new HashingEmbedder(64);
const usage = {
  inputTokens: { total: 1, noCache: 1, cacheRead: 0, cacheWrite: 0 },
  outputTokens: { total: 1, text: 1, reasoning: 0 },
};
const toolOpts = { toolCallId: "t", messages: [] } as never;

describe("asEmbedder", () => {
  it("wraps AI SDK embedding models", async () => {
    const model = new MockEmbeddingModelV4({
      doEmbed: async ({ values }) => ({ embeddings: values.map((v) => hashing.embed(v)), warnings: [] }),
    });
    const e = asEmbedder(model);
    expect(await e.embed("hello world")).toEqual(hashing.embed("hello world"));
    expect(await e.embedMany!(["a", "b"])).toHaveLength(2);
  });

  it("passes TideVec embedders and functions through", async () => {
    expect(asEmbedder(hashing)).toBe(hashing);
    expect(await asEmbedder((t: string) => [t.length]).embed("abc")).toEqual([3]);
  });
});

describe.skipIf(!hasServer)("with a live server", () => {
  let server: RunningServer;
  let client: TideVec;

  beforeAll(async () => {
    server = await startServer();
    client = new TideVec(server.url);
  });
  afterAll(async () => {
    await server?.stop();
  });

  it("memory tools supersede facts by key", async () => {
    const tools = tidevecMemoryTools({ client, embedding: hashing, collection: "ai_mem" });
    const first = await tools.remember.execute!(
      { text: "The user's favourite editor is Vim", key: "user.editor" },
      toolOpts,
    );
    const second = await tools.remember.execute!(
      { text: "The user switched to the Helix editor", key: "user.editor" },
      toolOpts,
    );
    expect(second).toMatchObject({ superseded: [(first as { id: string }).id] });

    const recalled = (await tools.recall.execute!({ query: "which editor does the user use" }, toolOpts)) as {
      text: string;
    }[];
    expect(recalled.map((m) => m.text)).toEqual(["The user switched to the Helix editor"]);

    const hist = (await tools.history.execute!({ key: "user.editor" }, toolOpts)) as unknown[];
    expect(hist).toHaveLength(2);
    expect(await tools.forget.execute!({ key: "user.editor" }, toolOpts)).toEqual({ deleted: 2 });
  });

  it("a model can call the recall tool through generateText", async () => {
    const tools = tidevecMemoryTools({ client, embedding: hashing, collection: "ai_mem_gen" });
    await tools.remember.execute!({ text: "The user lives in Lisbon", key: "user.city" }, toolOpts);

    let step = 0;
    const model = new MockLanguageModelV4({
      doGenerate: async () => {
        step++;
        if (step === 1) {
          return {
            content: [
              {
                type: "tool-call",
                toolCallId: "call-1",
                toolName: "recall",
                input: JSON.stringify({ query: "where does the user live" }),
              },
            ],
            finishReason: { unified: "tool-calls", raw: undefined },
            usage,
            warnings: [],
          };
        }
        return {
          content: [{ type: "text", text: "Lisbon" }],
          finishReason: { unified: "stop", raw: undefined },
          usage,
          warnings: [],
        };
      },
    });
    const result = await generateText({ model, tools, prompt: "Where do I live?", stopWhen: stepCountIs(3) });
    expect(result.text).toBe("Lisbon");
    const toolResult = result.steps[0]!.toolResults[0]!;
    expect(toolResult.toolName).toBe("recall");
    expect(JSON.stringify(toolResult.output)).toContain("Lisbon");
  });

  it("search tool and RAG middleware return only current document versions", async () => {
    const opts = { client, collection: "ai_kb", embedding: hashing };
    await addDocuments(
      { ...opts, createIfMissing: { dim: 64, matchField: "doc_id" } },
      [
        {
          id: "refunds-v1",
          text: "Refunds are processed within 14 days of the request",
          metadata: { doc_id: "refunds" },
          createdAt: Date.now() - 60 * DAY,
        },
      ],
    );
    await addDocuments(opts, [
      { id: "refunds-v2", text: "Refunds are processed within 3 days of the request", metadata: { doc_id: "refunds" } },
      { id: "shipping", text: "Shipping is free above 50 euros", metadata: { doc_id: "shipping" } },
    ]);

    const retrieve = createRetriever(opts);
    const docs = await retrieve("how long do refunds take", { topK: 2 });
    expect(docs.map((d) => d.id)).not.toContain("refunds-v1");
    expect(docs[0]!.id).toBe("refunds-v2");
    expect(docs[0]!.metadata).toEqual({ doc_id: "refunds" });

    const search = tidevecSearchTool(opts);
    const out = (await search.execute!({ query: "when are refunds processed", k: 1 }, toolOpts)) as { id: string }[];
    expect(out.map((d) => d.id)).toEqual(["refunds-v2"]);

    let seenPrompt: unknown;
    const base = new MockLanguageModelV4({
      doGenerate: async (params) => {
        seenPrompt = params.prompt;
        return {
          content: [{ type: "text", text: "3 days" }],
          finishReason: { unified: "stop", raw: undefined },
          usage,
          warnings: [],
        };
      },
    });
    const model = wrapLanguageModel({ model: base, middleware: tidevecRagMiddleware({ ...opts, topK: 2 }) });
    const r = await generateText({ model, prompt: "How long do refunds take?" });
    expect(r.text).toBe("3 days");
    const system = (seenPrompt as { role: string; content: unknown }[])[0]!;
    expect(system.role).toBe("system");
    expect(String(system.content)).toContain("within 3 days");
    expect(String(system.content)).not.toContain("14 days");
  });
});
