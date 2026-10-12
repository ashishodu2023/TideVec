import { describe, expect, it } from "vitest";
import {
  HashingEmbedder,
  NotFoundError,
  NotLeaderError,
  RateLimitError,
  TideVec,
  TideVecConnectionError,
  encodeFilter,
  normaliseUrl,
  payloadStr,
} from "../src/index.js";

function fakeFetch(handler: (url: string, init: RequestInit) => [number, unknown]) {
  const calls: { url: string; init: RequestInit }[] = [];
  const fn = async (url: string, init: RequestInit = {}) => {
    calls.push({ url, init });
    const [status, body] = handler(url, init);
    return new Response(typeof body === "string" ? body : JSON.stringify(body), { status });
  };
  return { fn, calls };
}

describe("helpers", () => {
  it("normalises urls like the Python client", () => {
    expect(normaliseUrl("localhost")).toBe("http://localhost:6399");
    expect(normaliseUrl("db.internal:7000")).toBe("http://db.internal:7000");
    expect(normaliseUrl("https://x.example.com/")).toBe("https://x.example.com");
    expect(normaliseUrl("[::1]:6399")).toBe("http://[::1]:6399");
  });

  it("encodes payload values and filters", () => {
    expect(payloadStr("a")).toBe("a");
    expect(payloadStr(3)).toBe("3");
    expect(payloadStr(["x"])).toBe('["x"]');
    expect(encodeFilter({ doc_id: "refunds" })).toBe("doc_id=refunds");
    expect(encodeFilter({ page: 3 })).toBe("page=3");
    expect(encodeFilter("a=b")).toBe("a=b");
    expect(encodeFilter(undefined)).toBeUndefined();
    const rich = { $and: [{ a: 1 }, { b: 2 }] };
    expect(encodeFilter(rich)).toBe(rich);
    expect(() => encodeFilter("nonsense")).toThrow(/key=value/);
  });

  it("hashing embedder is deterministic and normalised", () => {
    const e = new HashingEmbedder(64);
    const a = e.embed("refunds take 14 days");
    expect(a).toEqual(e.embed("refunds take 14 days"));
    expect(Math.hypot(...a)).toBeCloseTo(1, 6);
    expect(a).toHaveLength(64);
  });
});

describe("transport", () => {
  it("sends the api key and snake_case search body", async () => {
    const { fn, calls } = fakeFetch(() => [200, { status: "ok", data: { results: [], count: 0 } }]);
    const db = new TideVec({ url: "h:1", apiKey: "k", fetch: fn });
    await db.search("docs", [1, 0], { topK: 3, temporalBlend: 0, supersededPenalty: 1, filter: { a: "b" } });
    const { url, init } = calls[0]!;
    expect(url).toBe("http://h:1/v1/collections/docs/search");
    expect((init.headers as Record<string, string>)["X-Api-Key"]).toBe("k");
    expect(JSON.parse(init.body as string)).toEqual({
      vector: [1, 0],
      top_k: 3,
      temporal_blend: 0,
      superseded_penalty: 1,
      filter: "a=b",
    });
  });

  it("maps error statuses to typed errors", async () => {
    const { fn } = fakeFetch(() => [404, { status: "error", error: "Collection not found: x", code: 404 }]);
    const db = new TideVec({ fetch: fn });
    await expect(db.getCollection("x")).rejects.toBeInstanceOf(NotFoundError);
    expect(await db.hasCollection("x")).toBe(false);
    expect(await db.dropCollection("x")).toBe(false);
  });

  it("retries 429 and gives up after maxRetries", async () => {
    const { fn, calls } = fakeFetch(() => [429, { error: "slow down" }]);
    const db = new TideVec({ fetch: fn, maxRetries: 0 });
    await expect(db.info()).rejects.toBeInstanceOf(RateLimitError);
    expect(calls).toHaveLength(1);
  });

  it("follows Raft leader hints", async () => {
    const { fn, calls } = fakeFetch((url) =>
      url.startsWith("http://follower")
        ? [421, { error: "Not the Raft leader", leader: "http://leader:6399" }]
        : [201, { status: "ok", data: { inserted: 1, ids: ["a"] } }],
    );
    const db = new TideVec({ url: "http://follower:6399", fetch: fn });
    expect(await db.upsert("c", [{ id: "a", embedding: [1] }])).toBe(1);
    expect(calls.map((c) => c.url)).toEqual([
      "http://follower:6399/v1/collections/c/upsert",
      "http://leader:6399/v1/collections/c/upsert",
    ]);
  });

  it("raises NotLeaderError when no leader appears", async () => {
    const { fn } = fakeFetch(() => [421, { error: "Not the Raft leader", leader: null }]);
    const db = new TideVec({ fetch: fn, leaderWaitMs: 300 });
    const start = Date.now();
    const err = await db.upsert("c", [{ id: "a", embedding: [1] }]).catch((e) => e);
    expect(err).toBeInstanceOf(NotLeaderError);
    expect(err.leader).toBeNull();
    expect(Date.now() - start).toBeGreaterThanOrEqual(250);
  });

  it("wraps network failures", async () => {
    const db = new TideVec({
      fetch: async () => {
        throw new TypeError("fetch failed");
      },
    });
    await expect(db.health()).rejects.toBeInstanceOf(TideVecConnectionError);
    expect(await db.ping()).toBe(false);
  });

  it("maps vector fields to the wire format", async () => {
    const { fn, calls } = fakeFetch(() => [201, { status: "ok", data: { inserted: 1 } }]);
    const db = new TideVec({ fetch: fn });
    await db.upsert("c", [
      {
        id: "a",
        embedding: new Float32Array([0.5, 0.25]),
        payload: { n: 2, s: "x", skip: undefined },
        createdAt: new Date(1000),
        ttlSeconds: 60,
        edges: [{ targetId: "b", type: "CAUSES" }],
      },
    ]);
    expect(JSON.parse(calls[0]!.init.body as string)).toEqual({
      vectors: [
        {
          id: "a",
          embedding: [0.5, 0.25],
          payload: { n: "2", s: "x" },
          created_at: 1000,
          ttl_seconds: 60,
          edges: [{ target_id: "b", type: "CAUSES", weight: 1 }],
        },
      ],
    });
  });
});
