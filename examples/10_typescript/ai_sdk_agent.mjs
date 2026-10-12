// Vercel AI SDK agent with TideVec long-term memory and a supersession-aware knowledge tool.
//
//   npm install
//   OPENAI_API_KEY=... TIDEVEC_API_KEY=dev node ai_sdk_agent.mjs
import { openai } from "@ai-sdk/openai";
import { generateText, stepCountIs } from "ai";
import { addDocuments, MEMORY_INSTRUCTIONS, tidevecMemoryTools, tidevecSearchTool } from "@tidevec/ai-sdk";

const client = { url: process.env.TIDEVEC_URL ?? "localhost:6399" };
const embedding = openai.embedding("text-embedding-3-small");
const kb = { client, collection: "ts_agent_kb", embedding };

await addDocuments({ ...kb, createIfMissing: { dim: 1536, matchField: "doc_id" } }, [
  { id: "returns-2025", text: "Items can be returned within 14 days.", metadata: { doc_id: "returns" },
    createdAt: Date.now() - 200 * 86_400_000 },
  { id: "returns-2026", text: "Items can be returned within 30 days.", metadata: { doc_id: "returns" } },
]);

const tools = {
  ...tidevecMemoryTools({ client, embedding, collection: "ts_agent_memory" }),
  searchKnowledge: tidevecSearchTool(kb),
};

for (const prompt of [
  "Hi! I'm on the Pro plan, and I prefer short answers.",
  "Actually I upgraded to the Enterprise plan yesterday.",
  "Which plan am I on, and how long do I have to return an item?",
]) {
  const { text, steps } = await generateText({
    model: openai("gpt-4o-mini"),
    system: MEMORY_INSTRUCTIONS + "\nUse searchKnowledge for policy questions.",
    tools,
    stopWhen: stepCountIs(6),
    prompt,
  });
  const used = steps.flatMap((s) => s.toolCalls.map((c) => c.toolName));
  console.log(`> ${prompt}\n${text}\n  tools: ${used.join(", ") || "none"}\n`);
}
