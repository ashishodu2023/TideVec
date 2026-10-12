"""
Text embedders for the integrations, MCP server and examples.

Each embedder exposes ``dim``, ``embed_documents(texts)`` and
``embed_query(text)`` — the same method names LangChain uses, so any of
these can be passed wherever a LangChain ``Embeddings`` is expected.

    HashingEmbedder               no dependencies, lexical only (tests, demos)
    OpenAIEmbedder                OpenAI-compatible /embeddings endpoint
    SentenceTransformerEmbedder   local model via sentence-transformers
"""

from __future__ import annotations

import hashlib
import json as _json
import math
import os as _os
import re
import sys
import urllib.request
from typing import List, Optional, Sequence

_TOKEN = re.compile(r"\w+", re.UNICODE)


class HashingEmbedder:
    """
    Deterministic feature-hashing embedder over word unigrams and bigrams.

    No model and no network: near-identical texts get near-identical
    vectors, but there is no notion of synonyms. Good for tests, demos
    and offline CI; use a real model for production retrieval.
    """

    def __init__(self, dim: int = 384):
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _features(self, text: str) -> List[str]:
        words = _TOKEN.findall(text.lower())
        return words + [f"{a} {b}" for a, b in zip(words, words[1:])]

    def embed_query(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for feat in self._features(text):
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
            vec[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0 else vec

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]


class OpenAIEmbedder:
    """
    Embeddings from an OpenAI-compatible ``POST {base_url}/embeddings``.

    Args:
        model: e.g. "text-embedding-3-small" (1536-d) or "-large" (3072-d).
        dimensions: optional output size for text-embedding-3 models.
        api_key: defaults to $OPENAI_API_KEY.
        base_url: defaults to $OPENAI_BASE_URL or https://api.openai.com/v1.
    """

    _DEFAULT_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str = "text-embedding-3-small", *,
                 dimensions: Optional[int] = None, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, batch_size: int = 256,
                 timeout: float = 60.0):
        self.model = model
        self.dimensions = dimensions
        self.api_key = api_key or _os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError("OpenAIEmbedder needs api_key or $OPENAI_API_KEY")
        self.base_url = (base_url or _os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.dim = dimensions or self._DEFAULT_DIMS.get(model) or len(self.embed_query("dim probe"))

    def _call(self, texts: List[str]) -> List[List[float]]:
        body = {"model": self.model, "input": texts}
        if self.dimensions:
            body["dimensions"] = self.dimensions
        req = urllib.request.Request(
            f"{self.base_url}/embeddings", data=_json.dumps(body).encode(),
            method="POST", headers={"Content-Type": "application/json",
                                    "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = _json.loads(resp.read())["data"]
        return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        texts = list(texts)
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._call(texts[i:i + self.batch_size]))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self._call([text])[0]


class SentenceTransformerEmbedder:
    """Local embeddings via ``sentence-transformers`` (installed separately)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError("pip install sentence-transformers") from e
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._model.encode(list(texts), normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def default_embedder():
    """
    Best embedder available without configuration: sentence-transformers
    if installed, else OpenAI if $OPENAI_API_KEY is set, else hashing
    (with a warning, since it is lexical only).
    """
    try:
        return SentenceTransformerEmbedder()
    except ImportError:
        pass
    if _os.environ.get("OPENAI_API_KEY"):
        return OpenAIEmbedder()
    print("tidevec: no embedding model found (pip install sentence-transformers "
          "or set OPENAI_API_KEY); falling back to lexical HashingEmbedder",
          file=sys.stderr)
    return HashingEmbedder()
