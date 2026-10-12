import { afterAll, beforeAll, describe, expect, it } from "vitest";
import {
  AgentMemory,
  HashingEmbedder,
  NotFoundError,
  TideVec,
  UnauthorizedError,
} from "../src/index.js";
import { type RunningServer, hasServer, startServer } from "./server.js";

const DAY = 86_400_000;
const emb = new HashingEmbedder(64);

describe.skipIf(!hasServer)("live server", () => {
  let server: RunningServer;
  let db: TideVec;

  beforeAll(async () => {
    server = await startServer();
    db = new TideVec({ url: server.url, timeoutMs: 10_000 });
  });
  afterAll(async () => {
    await server?.stop();
  });

  it("reports health and info", async () => {
    const h = await db.health();
    expect(h.status).toBe("ok");
    expect(await db.ping()).toBe(true);
    expect((await db.info()).name).toBe("TideVec");
    expect(await db.metrics()).toContain("tidevec_requests_total");
    expect((await db.clusterStatus()).mode).toBe("standalone");
  });

  it("manages collections", async () => {
    expect(await db.hasCollection("c_admin")).toBe(false);
    expect(await db.createCollection("c_admin", { dim: 4, nShards: 2, metric: "cosine" })).toBe("c_admin");
    expect(await db.ensureCollection("c_admin", { dim: 4 })).toBe(false);
    const listed = await db.listCollections();
    expect(listed.find((c) => c.name === "c_admin")).toMatchObject({ dim: 4, nShards: 2 });
    const info = await db.getCollection("c_admin");
    expect(info.nVectors).toBe(0);
    await db.setTemporal("c_admin", { halfLifeMs: DAY, temporalBlend: 0.1 });
    expect(await db.dropCollection("c_admin")).toBe(true);
    expect(await db.dropCollection("c_admin")).toBe(false);
    await expect(db.getCollection("c_admin")).rejects.toBeInstanceOf(NotFoundError);
  });

  it("upserts, searches with temporal blend and filters, deletes", async () => {
    const col = db.collection("c_search");
    await db.createCollection(col.name, { dim: 4, temporal: { halfLifeMs: 7 * DAY, temporalBlend: 0.5 } });
    const now = Date.now();
    await col.upsert([
      { id: "old", embedding: [1, 0, 0, 0], payload: { topic: "a", n: 1 }, createdAt: now - 90 * DAY },
      { id: "new", embedding: [0.9, 0.1, 0, 0], payload: { topic: "a", n: 2 }, createdAt: now },
      { id: "other", embedding: [0, 0, 1, 0], payload: { topic: "b" } },
    ]);
    expect(await col.count()).toBe(3);

    const pure = await col.search([1, 0, 0, 0], { topK: 2, temporalBlend: 0 });
    expect(pure.hits[0]!.id).toBe("old");
    const blended = await col.search([1, 0, 0, 0], { topK: 3, temporalBlend: 0.5 });
    expect(blended.hits[0]!.id).toBe("new");
    expect(blended.hits[0]!.payload.n).toBe("2");
    const old = blended.hits.find((h) => h.id === "old")!;
    expect(blended.hits[0]!.temporalScore).toBeGreaterThan(old.temporalScore);

    const filtered = await col.search([1, 0, 0, 0], { topK: 5, filter: { topic: "b" } });
    expect(filtered.hits.map((h) => h.id)).toEqual(["other"]);

    const traced = await col.search([1, 0, 0, 0], { topK: 1, includeTrace: true });
    expect(typeof traced.trace?.latency_ms).toBe("number");

    expect(await col.delete({ filter: { topic: "b" } })).toBe(1);
    expect(await col.delete(["old", "missing"])).toBe(1);
    expect((await col.search([1, 0, 0, 0], { topK: 5 })).hits.map((h) => h.id)).toEqual(["new"]);
    await col.drop();
  });

  it("links superseded versions and can exclude them", async () => {
    const name = "c_super";
    await db.createCollection(name, {
      dim: 64,
      supersession: { similarityThreshold: 0.5, matchField: "doc_id" },
    });
    const now = Date.now();
    await db.upsert(name, [
      {
        id: "v1",
        embedding: emb.embed("Refunds are processed within 14 days of the request"),
        payload: { doc_id: "refunds" },
        createdAt: now - 60 * DAY,
      },
    ]);
    await db.upsert(name, [
      {
        id: "v2",
        embedding: emb.embed("Refunds are processed within 3 days of the request"),
        payload: { doc_id: "refunds" },
        createdAt: now,
      },
      { id: "ship", embedding: emb.embed("Shipping is free above 50 euros"), payload: { doc_id: "shipping" } },
    ]);
    const q = emb.embed("how long do refunds take");
    const all = await db.search(name, q, { topK: 3, temporalBlend: 0 });
    const v1 = all.hits.find((h) => h.id === "v1")!;
    expect(v1.supersededBy).toEqual(["v2"]);
    expect(v1.isSuperseded).toBe(true);

    const penalised = await db.search(name, q, { topK: 3, temporalBlend: 0, supersededPenalty: 1 });
    expect(penalised.hits.at(-1)!.id).toBe("v1");

    const current = await db.search(name, q, { topK: 2, excludeSuperseded: true });
    expect(current.hits.map((h) => h.id)).not.toContain("v1");
    expect(current.hits).toHaveLength(2);
    expect((await db.getCollection(name)).supersession).toMatchObject({ match_field: "doc_id" });
  });

  it("adds edges and expands causally", async () => {
    const name = "c_edges";
    await db.createCollection(name, { dim: 4 });
    await db.upsert(name, [
      { id: "cause", embedding: [1, 0, 0, 0] },
      { id: "effect", embedding: [0, 1, 0, 0], edges: [{ targetId: "cause", type: "RELATED_TO" }] },
    ]);
    expect(await db.addEdges(name, [{ src: "cause", tgt: "effect", type: "CAUSES", weight: 0.9 }])).toBe(1);
    const r = await db.search(name, [1, 0, 0, 0], { topK: 1, mode: "causal_expand", causalHops: 1 });
    expect(r.hits[0]!.id).toBe("cause");
    expect(r.hits[0]!.causalNeighbors).toContain("effect");
  });

  it("reports drift status and rejects disallowed re-embed urls", async () => {
    await db.createCollection("c_drift", { dim: 4 });
    const s = await db.driftStatus("c_drift");
    expect(s.phase).toBe("IDLE");
    await expect(
      db.startDrift("c_drift", { reembedUrl: "http://127.0.0.1:1/embed" }),
    ).rejects.toMatchObject({ status: 403 });
  });

  it("calls raw endpoints", async () => {
    const r = await db.request<{ status: string }>("GET", "/health");
    expect(r.status).toBe("ok");
    await expect(db.request("GET", "/v1/nope")).rejects.toBeInstanceOf(NotFoundError);
  });

  describe("retrieval-completeness endpoints", () => {
    let supported = false;
    beforeAll(async () => {
      await db.createCollection("c_p1", { dim: 4, textField: "text", supersession: { matchField: "doc" } });
      supported = (await db.getCollection("c_p1")).textField === "text";
      if (!supported) return;
      const t0 = Date.UTC(2025, 0, 1);
      await db.upsert("c_p1", [
        { id: "a", embedding: [1, 0, 0, 0], payload: { doc: "a", lang: "en", year: 2019, text: "error E1234 on boot" }, createdAt: t0 },
        { id: "b", embedding: [0, 1, 0, 0], payload: { doc: "b", lang: "de", year: 2021, text: "refunds take three days" }, createdAt: t0 + DAY },
        { id: "c", embedding: [0, 0, 1, 0], payload: { doc: "c", lang: "en", year: 2023, text: "shipping is free" }, createdAt: t0 + 2 * DAY },
      ]);
    });

    it("gets vectors by id", async (ctx) => {
      if (!supported) ctx.skip();
      const a = await db.get("c_p1", "a");
      expect(a).toMatchObject({ id: "a", payload: { lang: "en" }, embedding: [1, 0, 0, 0] });
      expect(await db.get("c_p1", "nope")).toBeNull();
      const many = await db.getMany("c_p1", ["c", "nope", "a"]);
      expect(many.vectors.map((v) => v.id)).toEqual(["c", "a"]);
      expect(many.missing).toEqual(["nope"]);
      expect(many.vectors[0]!.embedding).toBeUndefined();
      await expect(db.get("missing_collection", "a")).rejects.toBeInstanceOf(NotFoundError);
    });

    it("scrolls, lists ids and counts with structured filters", async (ctx) => {
      if (!supported) ctx.skip();
      const all: string[] = [];
      for await (const v of db.scroll("c_p1", { limit: 2 })) all.push(v.id);
      expect(all).toEqual(["a", "b", "c"]);
      const page = await db.scrollPage("c_p1", { limit: 2 });
      expect(page.items).toHaveLength(2);
      expect(page.nextCursor).not.toBeNull();

      const recentEn = { $and: [{ lang: "en" }, { year: { $gte: 2020 } }] };
      const ids: string[] = [];
      for await (const id of db.ids("c_p1", { filter: recentEn })) ids.push(id);
      expect(ids).toEqual(["c"]);
      expect(await db.count("c_p1", { filter: { lang: ["en", "de"] } })).toBe(3);
      expect(await db.count("c_p1", { filter: { $or: [{ lang: "de" }, { year: { $lt: 2020 } }] } })).toBe(2);
      expect(await db.count("c_p1", { asOf: new Date(Date.UTC(2025, 0, 1, 12)) })).toBe(1);

      const hits = await db.search("c_p1", [1, 1, 1, 0], { topK: 5, filter: { year: { $gt: 2020 } } });
      expect(hits.hits.map((h) => h.id).sort()).toEqual(["b", "c"]);
    });

    it("searches as of a past time", async (ctx) => {
      if (!supported) ctx.skip();
      const r = await db.search("c_p1", [0, 0, 1, 0], { topK: 3, asOf: "2025-01-02T12:00:00Z", temporalBlend: 0 });
      expect(r.hits.map((h) => h.id)).not.toContain("c");
    });

    it("runs keyword and hybrid search", async (ctx) => {
      if (!supported) ctx.skip();
      const kw = await db.search("c_p1", null, { text: "E1234", topK: 3 });
      expect(kw.hits[0]!.id).toBe("a");
      const hy = await db.search("c_p1", [0, 1, 0, 0], { text: "E1234", topK: 3, hybrid: { fusion: "rrf" } });
      expect(hy.hits.map((h) => h.id).slice(0, 2).sort()).toEqual(["a", "b"]);
      await expect(db.search("c_p1", null, {})).rejects.toThrow(/vector and\/or/);
    });

    it("manages aliases", async (ctx) => {
      if (!supported) ctx.skip();
      await db.setAlias("c_p1_live", "c_p1");
      expect(await db.listAliases()).toContainEqual({ alias: "c_p1_live", collection: "c_p1" });
      expect((await db.get("c_p1_live", "b"))?.id).toBe("b");
      expect((await db.getCollection("c_p1")).aliases).toContain("c_p1_live");
      expect(await db.deleteAlias("c_p1_live")).toBe(true);
      expect(await db.deleteAlias("c_p1_live")).toBe(false);
    });

    it("deletes by structured filter", async (ctx) => {
      if (!supported) ctx.skip();
      expect(await db.delete("c_p1", { filter: { year: { $lt: 2020 } } })).toBe(1);
      expect(await db.count("c_p1")).toBe(2);
    });
  });

  it("AgentMemory supersedes by key and keeps history", async () => {
    const mem = new AgentMemory(db, { embed: emb, collection: "c_memory" });
    const first = await mem.remember("User's favourite editor is Vim", {
      key: "user.editor",
      createdAt: Date.now() - DAY,
    });
    const second = await mem.remember("User switched to the Helix editor", { key: "user.editor" });
    expect(second.superseded).toEqual([first.id]);
    await mem.remember("User lives in Lisbon", { key: "user.city", metadata: { source: "chat", turn: 3 } });

    const recalled = await mem.recall("which editor does the user like?", { k: 3 });
    expect(recalled.map((m) => m.id)).toContain(second.id);
    expect(recalled.map((m) => m.id)).not.toContain(first.id);
    const city = recalled.find((m) => m.key === "user.city");
    if (city) expect(city.metadata).toEqual({ source: "chat", turn: "3" });

    const withOld = await mem.recall("editor", { k: 5, includeSuperseded: true, key: "user.editor" });
    expect(withOld.map((m) => m.id).sort()).toEqual([first.id, second.id].sort());

    const hist = await mem.history("user.editor");
    expect(hist.map((m) => m.id)).toEqual([first.id, second.id]);
    expect(hist[0]!.supersededBy).toEqual([second.id]);

    expect(await mem.forget({ key: "user.editor" })).toBe(2);
    expect(await mem.history("user.editor")).toEqual([]);
  });
});

describe.skipIf(!hasServer)("auth", () => {
  let server: RunningServer;
  beforeAll(async () => {
    server = await startServer(["--api-key", "secret-key", "--require-auth"]);
  });
  afterAll(async () => {
    await server?.stop();
  });

  it("requires X-Api-Key", async () => {
    await expect(new TideVec({ url: server.url, apiKey: "" }).listCollections()).rejects.toBeInstanceOf(
      UnauthorizedError,
    );
    await expect(new TideVec({ url: server.url, apiKey: "wrong" }).listCollections()).rejects.toBeInstanceOf(
      UnauthorizedError,
    );
    const ok = new TideVec({ url: server.url, apiKey: "secret-key" });
    expect(await ok.listCollections()).toEqual([]);
  });
});
