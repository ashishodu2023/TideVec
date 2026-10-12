// TideVec from TypeScript/JavaScript: supersession-aware search and agent memory.
//
//   docker run -p 6399:6399 -e TIDEVEC_API_KEY=dev averm004/tidevec:0.3.0
//   TIDEVEC_API_KEY=dev node quickstart.mjs            # TIDEVEC_URL defaults to localhost:6399
//
// Runs offline: HashingEmbedder is a lexical stand-in for a real embedding model.
import { AgentMemory, HashingEmbedder, HalfLife, TideVec } from "tidevec";

const DAY = 86_400_000;
const db = new TideVec({ url: process.env.TIDEVEC_URL ?? "localhost:6399" });
const emb = new HashingEmbedder(384);
const kb = `ts_quickstart_${Date.now()}`;

await db.createCollection(kb, {
  dim: 384,
  temporal: { halfLifeMs: HalfLife.ONE_MONTH, temporalBlend: 0.3 },
  supersession: { matchField: "doc_id", similarityThreshold: 0.5 },
});

const docs = [
  ["refunds-v1", "refunds", "Refunds are processed within 14 days.", 120],
  ["shipping", "shipping", "We ship to 40 countries with tracking.", 60],
  ["refunds-v2", "refunds", "Refunds are now processed within 3 days.", 1],
];
for (const [id, docId, text, ageDays] of docs) {
  await db.upsert(kb, [{ id, embedding: await emb.embed(text), payload: { doc_id: docId, text },
                         createdAt: Date.now() - ageDays * DAY }]);
}

const query = await emb.embed("how long do refunds take");
const plain = await db.search(kb, query, { topK: 3, temporalBlend: 0 });
console.log("plain similarity:");
for (const h of plain.hits) console.log(`  ${h.score.toFixed(3)} ${h.payload.text}${h.isSuperseded ? "  (superseded)" : ""}`);

const current = await db.search(kb, query, { topK: 2, excludeSuperseded: true });
console.log("TideVec (superseded versions dropped):");
for (const h of current.hits) console.log(`  ${h.score.toFixed(3)} ${h.payload.text}`);

console.log("count where doc_id=refunds:", await db.count(kb, { filter: { doc_id: "refunds" } }));

// Agent memory: same layout as Python AgentMemory and the tidevec-mcp server.
const memory = new AgentMemory(db, { embed: emb, collection: `ts_memory_${Date.now()}` });
await memory.remember("The user prefers dark mode", { key: "ui.theme" });
const { superseded } = await memory.remember("The user switched to light mode", { key: "ui.theme" });
console.log(`\nremember(ui.theme) superseded ${superseded.length} older memory`);
console.log("recall:", (await memory.recall("which theme does the user like")).map((m) => m.text));
console.log("history:", (await memory.history("ui.theme")).map((m) => m.text));

await db.dropCollection(kb);
await db.dropCollection(memory.collection);
