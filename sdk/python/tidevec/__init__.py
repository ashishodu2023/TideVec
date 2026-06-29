"""TideVec Python SDK — Temporally-aware causal vector database."""

from .client import (
    TideVec,
    AsyncTideVec,
    SearchHit,
    SearchResponse,
    CollectionInfo,
    HalfLife,
)

__version__ = "0.1.0"
__all__ = [
    "TideVec",
    "AsyncTideVec",
    "SearchHit",
    "SearchResponse",
    "CollectionInfo",
    "HalfLife",
]
