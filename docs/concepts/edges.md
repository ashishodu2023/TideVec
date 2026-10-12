# Edges and contradictions

TideVec stores typed, weighted edges between vectors in the same collection, next to the vectors
themselves (per shard, persisted with them).

| Type | Meaning | Who creates it |
|---|---|---|
| `UPDATES` | newer version → older version | [supersession](supersession.md), or you |
| `CONTRADICTS` | the two cannot both be true (stored in both directions) | you, or the LLM judge below |
| `SUPPORTS` | the source confirms the target | you, or the LLM judge below |
| `CAUSES` | cause → effect | you |
| `ENTITY_OF` | mention → entity | you |
| `RELATED_TO` | generic relation | you |

## Adding edges

```python
db.add_edges("kb", [
    {"src": "incident-42", "tgt": "deploy-17", "type": "CAUSES", "weight": 0.9},
])
```

Or inline when upserting, with `edges: [{"target_id": ..., "type": ..., "weight": ...}]` on each
vector. REST: [`POST /v1/collections/{name}/edges`](../api/rest.md#post-v1-collections-name-edges).

## Using edges in search

`mode` selects how a search uses the graph:

- `vector_only` (default): plain ranked search; `superseded_by` is still reported.
- `contradiction_check`: fills `hit.contradicted_by` from `CONTRADICTS` edges, so the caller can
  warn that sources disagree or show both sides.
- `causal_expand`: adds neighbours reachable within `causal_hops` (default 1) to `hit.causal_neighbors`.

```python
resp = db.search("kb", q, top_k=5, mode="contradiction_check")
for hit in resp:
    if hit.contradicted_by:
        print(hit.id, "is contradicted by", hit.contradicted_by)
```

## Contradictions

TideVec does not decide on its own that two documents contradict each other. Supersession handles
"same document, newer version"; deciding whether two *different* sources disagree needs judgement.
`tidevec.contradictions.detect_contradictions` asks an LLM judge about the new document's nearest
neighbours and records the verdict as an edge:

```python
from tidevec.contradictions import detect_contradictions, openai_judge

edges = detect_contradictions(
    db, "kb", new_id, new_text, new_embedding,
    judge=openai_judge("gpt-4o-mini"),   # any OpenAI-compatible endpoint (OPENAI_BASE_URL)
    text_key="text", k=5, min_similarity=0.75,
)
# -> [{"tgt": "policy-2024", "type": "CONTRADICTS"}, ...]
```

A judge is any function `judge(new_text, existing_text) -> "contradicts" | "supports" | "unrelated"`,
so you can plug in a local model or rules. Neighbours below `min_similarity` (`vector_score`; 0.75 is
cosine 0.5) are not sent to the judge. Each call costs one LLM request per neighbour judged.

[`AgentMemory`](../integrations/mcp.md) accepts the same judge as `contradiction_judge=` and runs it on
every `remember()`.
