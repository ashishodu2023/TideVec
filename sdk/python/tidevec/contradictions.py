"""
Contradiction detection with an LLM judge.

TideVec links *versions* of the same fact automatically (supersession).
Two different sources that disagree are a different relation: they need
a judgement call, so this module asks a model and records the answer as
a CONTRADICTS or SUPPORTS edge. Searches then report contradicting hits
via ``hit.contradicted_by`` (and ``mode="contradiction_check"``).

    from tidevec.contradictions import detect_contradictions, openai_judge

    detect_contradictions(db, "kb", new_id, new_text, new_embedding,
                          judge=openai_judge(), text_key="text")
"""

from __future__ import annotations

import json as _json
import os as _os
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence

Judge = Callable[[str, str], str]
"""judge(new_text, existing_text) -> "contradicts" | "supports" | "unrelated"."""

_VERDICTS = {"contradicts": "CONTRADICTS", "supports": "SUPPORTS"}

_PROMPT = (
    "You compare two statements from a knowledge base.\n"
    "Statement A (new): {a}\n"
    "Statement B (existing): {b}\n"
    "Answer with exactly one word: 'contradicts' if both cannot be true at the "
    "same time, 'supports' if A confirms B, otherwise 'unrelated'."
)


def openai_judge(model: str = "gpt-4o-mini", *, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, timeout: float = 30.0) -> Judge:
    """A judge backed by an OpenAI-compatible chat completions endpoint."""
    key = api_key or _os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("openai_judge needs api_key or $OPENAI_API_KEY")
    url = (base_url or _os.environ.get("OPENAI_BASE_URL")
           or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"

    def judge(a: str, b: str) -> str:
        body = {"model": model, "temperature": 0,
                "messages": [{"role": "user", "content": _PROMPT.format(a=a, b=b)}]}
        req = urllib.request.Request(url, data=_json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = _json.loads(resp.read())["choices"][0]["message"]["content"]
        return text.strip().lower().strip(".'\"")

    return judge


def detect_contradictions(db: Any, collection: str, new_id: str, new_text: str,
                          embedding: Sequence[float], judge: Judge, *,
                          text_key: str = "text", k: int = 5,
                          min_similarity: float = 0.75,
                          skip_ids: Sequence[str] = ()) -> List[Dict[str, str]]:
    """
    Judge ``new_text`` against its ``k`` nearest neighbours whose
    ``vector_score`` ((1 + cos) / 2 for cosine; 0.75 = cos 0.5) is at least
    ``min_similarity``, and add CONTRADICTS / SUPPORTS edges from the new
    vector. Returns the edges added as ``{"tgt", "type"}`` dicts.
    """
    skip = set(skip_ids) | {new_id}
    resp = db.search(collection, list(embedding), top_k=k + len(skip),
                     temporal_blend=0.0, include_staleness_warnings=False)
    edges = []
    for hit in resp.hits:
        if hit.id in skip or hit.vector_score < min_similarity:
            continue
        other = hit.payload.get(text_key, "")
        if not other:
            continue
        words = judge(new_text, other).split()
        verdict = _VERDICTS.get(words[0] if words else "")
        if verdict:
            edges.append({"src": new_id, "tgt": hit.id, "type": verdict, "weight": 1.0})
        if len(edges) >= k:
            break
    if edges:
        db.add_edges(collection, edges)
    return [{"tgt": e["tgt"], "type": e["type"]} for e in edges]
