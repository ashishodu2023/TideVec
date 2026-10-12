"""TideVec Python SDK — Temporally-aware causal vector database."""

from .client import (
    TideVec,
    AsyncTideVec,
    SearchHit,
    SearchResponse,
    Record,
    ScrollPage,
    CollectionInfo,
    CompactionStatus,
    DriftStatus,
    HalfLife,
    TideVecError,
    UnauthorizedError,
    ForbiddenError,
    RateLimitError,
    NotLeaderError,
    CollectionNotFoundError,
    APIError,
    payload_str,
)
from .local import Local

__version__ = "0.3.0"
__all__ = [
    "TideVec",
    "AsyncTideVec",
    "Local",
    "SearchHit",
    "SearchResponse",
    "Record",
    "ScrollPage",
    "CollectionInfo",
    "CompactionStatus",
    "DriftStatus",
    "HalfLife",
    "TideVecError",
    "UnauthorizedError",
    "ForbiddenError",
    "RateLimitError",
    "NotLeaderError",
    "CollectionNotFoundError",
    "APIError",
    "payload_str",
]
