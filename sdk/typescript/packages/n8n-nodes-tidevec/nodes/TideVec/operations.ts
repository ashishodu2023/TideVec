// n8n-independent core of the node, so it can be tested against a live
// server without an n8n runtime.
import type { TideVec, VectorInput } from "tidevec";

export type Resource = "collection" | "vector";
export type Operation =
  | "create"
  | "get"
  | "list"
  | "delete"
  | "upsert"
  | "search"
  | "deleteVectors";

export interface OperationParams {
  collection?: string;
  // collection.create
  dim?: number;
  metric?: string;
  halfLifeDays?: number;
  temporalBlend?: number;
  supersession?: boolean;
  supersedeMatchField?: string;
  supersedeThreshold?: number;
  // vector.upsert
  id?: string;
  embedding?: unknown;
  payload?: unknown;
  createdAt?: string | number;
  // vector.search
  topK?: number;
  excludeSuperseded?: boolean;
  supersededPenalty?: number;
  filter?: string;
  // vector.deleteVectors
  ids?: string;
}

const DAY_MS = 86_400_000;

export function parseEmbedding(value: unknown): number[] {
  const arr = typeof value === "string" ? JSON.parse(value) : value;
  if (!Array.isArray(arr) || arr.length === 0 || !arr.every((x) => typeof x === "number" && Number.isFinite(x))) {
    throw new Error("Embedding must be a non-empty array of numbers");
  }
  return arr as number[];
}

export function parsePayload(value: unknown): Record<string, unknown> {
  if (value === undefined || value === null || value === "") return {};
  const obj = typeof value === "string" ? JSON.parse(value) : value;
  if (typeof obj !== "object" || Array.isArray(obj)) throw new Error("Payload must be a JSON object");
  return obj as Record<string, unknown>;
}

function parseTimestamp(value: string | number | undefined): number | undefined {
  if (value === undefined || value === "") return undefined;
  if (typeof value === "number") return value;
  if (/^\d+$/.test(value)) return Number(value);
  const t = Date.parse(value);
  if (Number.isNaN(t)) throw new Error(`Invalid timestamp: ${value}`);
  return t;
}

function need<T>(value: T | undefined | "", name: string): T {
  if (value === undefined || value === "") throw new Error(`Parameter "${name}" is required`);
  return value as T;
}

/** Run one node operation. Returns the JSON items to emit. */
export async function executeOperation(
  client: TideVec,
  resource: Resource,
  operation: Operation,
  p: OperationParams,
): Promise<Record<string, unknown>[]> {
  if (resource === "collection") {
    switch (operation) {
      case "create": {
        const name = need(p.collection, "collection");
        const created = await client.ensureCollection(name, {
          dim: need(p.dim, "dim"),
          metric: p.metric,
          temporal: {
            halfLifeMs: p.halfLifeDays !== undefined ? Math.round(p.halfLifeDays * DAY_MS) : undefined,
            temporalBlend: p.temporalBlend,
          },
          supersession: p.supersession
            ? {
                matchField: p.supersedeMatchField || undefined,
                similarityThreshold: p.supersedeThreshold,
              }
            : undefined,
        });
        return [{ name, created }];
      }
      case "get":
        return [{ ...(await client.getCollection(need(p.collection, "collection"))) }];
      case "list":
        return (await client.listCollections()).map((c) => ({ ...c }));
      case "delete": {
        const name = need(p.collection, "collection");
        return [{ name, deleted: await client.dropCollection(name) }];
      }
    }
  }
  if (resource === "vector") {
    const collection = need(p.collection, "collection");
    switch (operation) {
      case "upsert": {
        const vector: VectorInput = {
          id: need(p.id, "id"),
          embedding: parseEmbedding(p.embedding),
          payload: parsePayload(p.payload),
          createdAt: parseTimestamp(p.createdAt),
        };
        const inserted = await client.upsert(collection, [vector]);
        return [{ id: vector.id, inserted }];
      }
      case "search": {
        const { hits } = await client.search(collection, parseEmbedding(p.embedding), {
          topK: p.topK,
          temporalBlend: p.temporalBlend,
          excludeSuperseded: p.excludeSuperseded,
          supersededPenalty: p.supersededPenalty,
          filter: p.filter || undefined,
        });
        return hits.map((h, rank) => ({ rank: rank + 1, ...h, createdAt: new Date(h.createdAt).toISOString() }));
      }
      case "deleteVectors": {
        const ids = (p.ids ?? "")
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean);
        if (!ids.length && !p.filter) throw new Error('Pass "ids" or "filter"');
        const deleted = await client.delete(collection, {
          ids: ids.length ? ids : undefined,
          filter: p.filter || undefined,
        });
        return [{ deleted }];
      }
    }
  }
  throw new Error(`Unsupported operation ${resource}.${operation}`);
}
