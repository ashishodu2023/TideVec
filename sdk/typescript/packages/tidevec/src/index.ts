export {
  TideVec,
  Collection,
  payloadStr,
  encodeFilter,
  normaliseUrl,
  type TideVecOptions,
  type RequestOptions,
  type DeleteOptions,
  type CountOptions,
  type FetchLike,
} from "./client.js";
export {
  TideVecError,
  TideVecConnectionError,
  APIError,
  NotFoundError,
  UnauthorizedError,
  ForbiddenError,
  RateLimitError,
  NotLeaderError,
} from "./errors.js";
export { AgentMemory, type AgentMemoryOptions, type Memory, type RecallOptions, type RememberOptions } from "./memory.js";
export { HashingEmbedder, toEmbedder, embedAll, type Embedder, type EmbedFn } from "./embeddings.js";
export * from "./types.js";

export const VERSION = "0.3.0";
