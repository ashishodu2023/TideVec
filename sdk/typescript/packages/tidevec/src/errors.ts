/** Base class for every error thrown by the client. */
export class TideVecError extends Error {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message, options);
    this.name = new.target.name;
  }
}

/** The server could not be reached (DNS, refused connection, timeout). */
export class TideVecConnectionError extends TideVecError {}

/** Any non-2xx response from the server. */
export class APIError extends TideVecError {
  constructor(
    message: string,
    readonly status: number,
    readonly body?: unknown,
  ) {
    super(`[${status}] ${message}`);
  }
}

/** HTTP 404: unknown collection, vector or endpoint. */
export class NotFoundError extends APIError {}

/** HTTP 401: X-Api-Key missing or invalid. */
export class UnauthorizedError extends APIError {}

/** HTTP 403: the tenant may not access this collection, or a quota is exceeded. */
export class ForbiddenError extends APIError {}

/** HTTP 429: rate limit exceeded (after the client's own retries). */
export class RateLimitError extends APIError {}

/**
 * HTTP 421: a write reached a Raft follower (experimental cluster mode).
 * The client follows the leader hint automatically; this is only thrown
 * when no leader emerged within the retry budget.
 */
export class NotLeaderError extends APIError {
  constructor(
    message: string,
    status: number,
    readonly leader: string | null,
    body?: unknown,
  ) {
    super(message, status, body);
  }
}

export function errorFromResponse(status: number, body: unknown): APIError {
  const obj = (body && typeof body === "object" ? body : {}) as Record<string, unknown>;
  const message =
    typeof obj.error === "string"
      ? obj.error
      : typeof body === "string" && body
        ? body
        : `HTTP ${status}`;
  switch (status) {
    case 401:
      return new UnauthorizedError(message, status, body);
    case 403:
      return new ForbiddenError(message, status, body);
    case 404:
      return new NotFoundError(message, status, body);
    case 421:
      return new NotLeaderError(
        message,
        status,
        typeof obj.leader === "string" ? obj.leader : null,
        body,
      );
    case 429:
      return new RateLimitError(message, status, body);
    default:
      return new APIError(message, status, body);
  }
}
