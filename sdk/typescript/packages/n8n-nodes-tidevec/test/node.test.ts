import type { IExecuteFunctions, INodeExecutionData } from "n8n-workflow";
import { HashingEmbedder, TideVec as TideVecClient } from "tidevec";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { type RunningServer, hasServer, startServer } from "../../tidevec/test/server.js";
import { TideVecApi } from "../credentials/TideVecApi.credentials";
import { TideVec } from "../nodes/TideVec/TideVec.node";
import { executeOperation, parseEmbedding, parsePayload } from "../nodes/TideVec/operations";

const DAY = 86_400_000;
const emb = new HashingEmbedder(32);

/** Minimal IExecuteFunctions: one parameter set per input item. */
function fakeContext(url: string, params: Record<string, unknown>[], continueOnFail = false) {
  return {
    getCredentials: async () => ({ url, apiKey: "" }),
    getInputData: () => params.map(() => ({ json: {} })),
    getNodeParameter: (name: string, i: number, fallback?: unknown) => params[i]![name] ?? fallback,
    continueOnFail: () => continueOnFail,
    getNode: () => ({ name: "TideVec", type: "tideVec", typeVersion: 1, position: [0, 0], parameters: {} }),
  } as unknown as IExecuteFunctions;
}

async function run(url: string, params: Record<string, unknown>[], continueOnFail = false) {
  const [out] = (await new TideVec().execute.call(fakeContext(url, params, continueOnFail))) as INodeExecutionData[][];
  return out!.map((o) => o.json);
}

describe("node description", () => {
  it("is consistent", () => {
    const d = new TideVec().description;
    expect(d.name).toBe("tideVec");
    expect(d.credentials?.[0]?.name).toBe(new TideVecApi().name);
    const ops = d.properties.filter((p) => p.name === "operation").flatMap((p) => p.options ?? []);
    expect(ops.map((o) => ("value" in o ? o.value : undefined))).toEqual(
      expect.arrayContaining(["create", "get", "list", "delete", "upsert", "search", "deleteVectors"]),
    );
  });

  it("parses inputs", () => {
    expect(parseEmbedding("[1, 2.5]")).toEqual([1, 2.5]);
    expect(() => parseEmbedding([])).toThrow();
    expect(() => parseEmbedding(["a"])).toThrow();
    expect(parsePayload('{"a": 1}')).toEqual({ a: 1 });
    expect(parsePayload("")).toEqual({});
    expect(() => parsePayload("[1]")).toThrow();
  });
});

describe.skipIf(!hasServer)("against a live server", () => {
  let server: RunningServer;
  beforeAll(async () => {
    server = await startServer();
  });
  afterAll(async () => {
    await server?.stop();
  });

  it("runs every operation through executeOperation", async () => {
    const client = new TideVecClient(server.url);
    const created = await executeOperation(client, "collection", "create", {
      collection: "n8n_ops",
      dim: 32,
      halfLifeDays: 7,
      supersession: true,
      supersedeMatchField: "doc_id",
      supersedeThreshold: 0.5,
    });
    expect(created).toEqual([{ name: "n8n_ops", created: true }]);

    await executeOperation(client, "vector", "upsert", {
      collection: "n8n_ops",
      id: "v1",
      embedding: emb.embed("Refunds are processed within 14 days of the request"),
      payload: '{"doc_id": "refunds"}',
      createdAt: new Date(Date.now() - 60 * DAY).toISOString(),
    });
    await executeOperation(client, "vector", "upsert", {
      collection: "n8n_ops",
      id: "v2",
      embedding: JSON.stringify(emb.embed("Refunds are processed within 3 days of the request")),
      payload: { doc_id: "refunds" },
    });

    const all = await executeOperation(client, "vector", "search", {
      collection: "n8n_ops",
      embedding: emb.embed("how long are refunds processed"),
      topK: 5,
      temporalBlend: 0,
    });
    expect(all.find((h) => h.id === "v1")?.supersededBy).toEqual(["v2"]);
    const current = await executeOperation(client, "vector", "search", {
      collection: "n8n_ops",
      embedding: emb.embed("how long are refunds processed"),
      topK: 5,
      excludeSuperseded: true,
    });
    expect(current.map((h) => h.id)).toEqual(["v2"]);

    expect((await executeOperation(client, "collection", "get", { collection: "n8n_ops" }))[0]).toMatchObject({
      nVectors: 2,
    });
    expect((await executeOperation(client, "collection", "list", {})).map((c) => c.name)).toContain("n8n_ops");
    expect(
      await executeOperation(client, "vector", "deleteVectors", { collection: "n8n_ops", filter: "doc_id=refunds" }),
    ).toEqual([{ deleted: 2 }]);
    expect(await executeOperation(client, "collection", "delete", { collection: "n8n_ops" })).toEqual([
      { name: "n8n_ops", deleted: true },
    ]);
  });

  it("executes as an n8n node over several items", async () => {
    await run(server.url, [
      { resource: "collection", operation: "create", collection: "n8n_node", dim: 32, metric: "cosine", halfLifeDays: 30, supersession: false },
    ]);
    const upserted = await run(server.url, [
      { resource: "vector", operation: "upsert", collection: "n8n_node", id: "a", embedding: emb.embed("alpha beta"), payload: '{"k": "x"}' },
      { resource: "vector", operation: "upsert", collection: "n8n_node", id: "b", embedding: emb.embed("gamma delta"), payload: '{"k": "y"}' },
    ]);
    expect(upserted).toEqual([
      { id: "a", inserted: 1 },
      { id: "b", inserted: 1 },
    ]);
    const hits = await run(server.url, [
      { resource: "vector", operation: "search", collection: "n8n_node", embedding: emb.embed("alpha beta"), topK: 1, excludeSuperseded: true, filter: "", options: { temporalBlend: 0 } },
    ]);
    expect(hits).toHaveLength(1);
    expect(hits[0]).toMatchObject({ rank: 1, id: "a", payload: { k: "x" } });

    const failed = await run(
      server.url,
      [{ resource: "vector", operation: "search", collection: "missing_col", embedding: [1], topK: 1 }],
      true,
    );
    expect(String(failed[0]!.error)).toMatch(/not found/i);
  });
});
