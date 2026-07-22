#!/usr/bin/env python3
"""
TideVec vs Qdrant — Temporal Ranking Benchmark
==============================================

Answers the reviewer question:
  "How does TideVec compare to Qdrant with a timestamp metadata filter?"

Corpus
------
Synthetic e-commerce refund / return policy knowledge base with 64 documents:
  · 8 policy topics × 4 dated versions each  (32 policy docs)
  · 20 related FAQ / support distractors
  · 12 off-topic distractors (shipping, account, billing)

The critical failure mode is intentional: for each topic the *stale* version is
constructed with a *higher* semantic similarity to the query than the current
version (mirroring the live demo where policy_v1 scores 0.837 vs policy_v2 at
0.822). Pure cosine therefore prefers outdated policy text.

Strategies compared
-------------------
1. Qdrant cosine-only          — default ANN, no time awareness
2. Qdrant + hard filter ≤14d   — timestamp metadata filter (common pattern)
3. Qdrant + hard filter ≤30d   — looser window
4. TideVec temporal blend      — α·cosine + β·2^(−age / half_life)

No external deps required (stdlib only). Optional: --qdrant-url to validate the
hard-filter path against a live Qdrant instance (same scoring math either way).

Usage
-----
  python3 run_benchmark.py
  python3 run_benchmark.py --half-life-days 7 --blend 0.3 --json results/out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Embeddings — deterministic hashed bag-of-words (no ML deps)
# ---------------------------------------------------------------------------

DIM = 128
MS_PER_DAY = 86_400_000


def _tokenize(text: str) -> List[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def embed(text: str, dim: int = DIM) -> List[float]:
    """Stable signed hashed BoW → L2-normalised vector."""
    vec = [0.0] * dim
    for tok in _tokenize(text):
        h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        # Longer / repeated tokens boost magnitude — used to control stale bias
        vec[idx] += sign * (1.0 + 0.15 * len(tok))
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def temporal_weight(age_days: float, half_life_days: float) -> float:
    """Ebbinghaus: 2^(−age / half_life) — matches include/tidevec/core/temporal_scorer.hpp."""
    if age_days <= 0:
        return 1.0
    return 2.0 ** (-age_days / half_life_days)


def blend_score(semantic: float, age_days: float, half_life: float, beta: float) -> float:
    return (1.0 - beta) * semantic + beta * temporal_weight(age_days, half_life)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

@dataclass
class Doc:
    id: str
    topic: str
    text: str
    age_days: int
    is_current: bool
    category: str  # policy | faq | distractor
    version: int = 0
    embedding: List[float] = field(default_factory=list)


@dataclass
class Query:
    id: str
    text: str
    topic: str
    gold_id: str  # current correct policy doc id
    stale_ids: List[str]  # outdated versions that must NOT rank #1


# Policy topics: (topic_key, query, versions[(age_days, is_current, text_body)])
#
# Age design (the hard-filter trap):
#   · Group A — current policy is fresh (≤7d): filter≤14d and TideVec both succeed
#   · Group B — current policy is 18–25d old (still authoritative, just not rewritten):
#       filter≤14d EXCLUDES the gold doc; a stale v3 at 20–28d may sit inside ≤30d
#       and beat gold on cosine. TideVec soft-decay still ranks gold #1.
_POLICY_SPECS = [
    # --- Group A: fresh current -------------------------------------------------
    (
        "electronics_refund",
        "What is the refund window for electronics?",
        [
            (90, False, "Electronics Refund Policy v1 (2023): Customers may return any electronic device within sixty (60) days of purchase for a full refund. Opened packaging is accepted. Original receipt required. Extended holiday returns apply November through January."),
            (45, False, "Electronics Refund Policy v2 (2024-Q1): Electronics returns accepted within forty-five (45) days. Device must power on. Packaging should be included when available. Proof of purchase mandatory. Holiday extension through January 15."),
            (21, False, "Electronics Refund Policy v3 (2024-Q3): Refund window for electronics is thirty (30) days from delivery. Device must be in working condition. Box preferred but not required. Receipt or order ID needed."),
            (3, True, "Electronics Refund Policy v4 (current): Electronics may be returned within 14 days of delivery for a full refund. Device must power on and include all accessories. Order confirmation required."),
        ],
    ),
    (
        "software_return",
        "Can I return opened software?",
        [
            (120, False, "Software Returns Policy v1: Physical software media may be returned within 30 days even if the seal is broken, provided the license key has not been activated online. Digital downloads are non-refundable after download completes."),
            (60, False, "Software Returns Policy v2: Opened software boxes are returnable within 21 days if the product key remains unused. Downloaded software refunds require a support ticket within 14 days of purchase."),
            (28, False, "Software Returns Policy v3: Unopened software: 30-day return. Opened physical media: 14-day return if unused key. Digital licenses: refund only if not redeemed, within 7 days."),
            (5, True, "Software Returns Policy v4 (current): Opened software is non-refundable once the license key is revealed or redeemed. Unopened boxed software: 14-day return. Digital purchases: no refund after download."),
        ],
    ),
    (
        "shipping_refund",
        "How long do shipping fee refunds take?",
        [
            (75, False, "Shipping Fee Refund Policy v1: Shipping charges are refunded within 10–14 business days after the returned item is scanned at our warehouse. Expedited shipping fees are non-refundable under any circumstance."),
            (40, False, "Shipping Fee Refund Policy v2: Standard shipping fees refunded in 7–10 business days post-receipt. Expedited fees refunded only if we shipped the wrong item. International duties never refunded."),
            (18, False, "Shipping Fee Refund Policy v3: Shipping refunds post within 5–7 business days of warehouse receipt. Expedited shipping refunded for carrier loss only."),
            (2, True, "Shipping Fee Refund Policy v4 (current): Shipping fee refunds appear in 3–5 business days after we receive the return. Expedited and overnight fees are non-refundable except for our fulfillment errors."),
        ],
    ),
    (
        "digital_goods",
        "Are digital gift cards refundable?",
        [
            (85, False, "Digital Goods Refund Policy v1: Gift cards and digital codes are refundable within 48 hours of purchase if unused. After 48 hours, balance transfers only. No cash refunds on promotional cards."),
            (35, False, "Digital Goods Refund Policy v2: Unused digital gift cards refundable within 24 hours. Partial-balance cards non-refundable. Promotional cards never refundable."),
            (16, False, "Digital Goods Refund Policy v3: Digital gift cards: refund within 12 hours if balance unused. No refunds after first redemption."),
            (1, True, "Digital Goods Refund Policy v4 (current): Digital gift cards and download codes are non-refundable once purchased. Unused physical gift cards may be returned within 14 days with receipt."),
        ],
    ),
    # --- Group B: authoritative-but-not-fresh current (filter≤14d trap) ---------
    (
        "apparel_exchange",
        "What is the exchange policy for clothing?",
        [
            (100, False, "Apparel Exchange Policy v1: Clothing may be exchanged within 90 days for any reason. Tags must remain attached. Worn or washed items accepted for store credit only. Final-sale clearance excluded."),
            (55, False, "Apparel Exchange Policy v2: Exchanges within 60 days with tags attached. Washed items eligible for store credit. Intimate apparel final sale. Free return shipping on exchanges."),
            (26, False, "Apparel Exchange Policy v3: Clothing exchanges within 45 days, tags on. Store credit for worn items. Free return label for size exchanges."),
            (22, True, "Apparel Exchange Policy v4 (current): Apparel exchanges within 30 days with original tags. Items washed or worn are not eligible. Free prepaid label for size/color exchanges only."),
        ],
    ),
    (
        "warranty_claim",
        "How do I file a product warranty claim?",
        [
            (110, False, "Warranty Claims Policy v1: File claims by emailing warranty@example.com with order ID, photos, and serial number. Coverage is 24 months from purchase. Claims processed in 15 business days. Refurbished replacements only."),
            (50, False, "Warranty Claims Policy v2: Submit claims via account portal → Orders → Warranty. 18-month coverage. Photo evidence required. Processing in 10 business days. New or refurbished at our discretion."),
            (28, False, "Warranty Claims Policy v3: Warranty claims via Help Center form. 12-month coverage from delivery. Serial + photos required. 7 business day review."),
            (20, True, "Warranty Claims Policy v4 (current): File warranty claims in the app under Orders → Report issue. Coverage is 12 months from delivery. Include serial number and 3 photos. Decisions within 5 business days; replacements are new units when stock allows."),
        ],
    ),
    (
        "subscription_cancel",
        "How do I cancel my subscription and get a refund?",
        [
            (95, False, "Subscription Cancellation Policy v1: Cancel anytime in Account → Billing. Full refund of the current billing cycle if cancelled within 14 days of charge. Annual plans prorated monthly. Contact support for enterprise plans."),
            (42, False, "Subscription Cancellation Policy v2: Cancel in Account → Billing. Refund within 7 days of renewal charge. Annual plans: unused months as account credit. Chat support for business tiers."),
            (25, False, "Subscription Cancellation Policy v3: Self-serve cancel in Billing. Refund window 5 days after renewal. Annual: store credit for unused months."),
            (18, True, "Subscription Cancellation Policy v4 (current): Cancel under Account → Billing. Full refund only if cancelled within 3 days of the renewal charge. Annual plans convert unused time to store credit; no cash refund after day 3."),
        ],
    ),
    (
        "damaged_item",
        "What if my order arrives damaged?",
        [
            (70, False, "Damaged Item Policy v1: Report damage within 14 days of delivery with photos. We ship a free replacement or issue a full refund including shipping. Keep the damaged item until the claim is closed."),
            (38, False, "Damaged Item Policy v2: Report within 10 days with unboxing photos. Free replacement or refund of item price. Retain packaging for carrier inspection."),
            (27, False, "Damaged Item Policy v3: Report damage within 7 days via Orders → Report issue. Replacement or refund. Photos of box and product required."),
            (19, True, "Damaged Item Policy v4 (current): Report damaged deliveries within 48 hours in the app with photos of the box and item. We replace at no cost or refund the item; original packaging must be kept for 7 days for carrier claim."),
        ],
    ),
]


_FAQ_TEXTS = [
    ("faq_track_return", "faq", 10, "How do I track my return shipment? Use the prepaid label barcode on the Returns page."),
    ("faq_restocking", "faq", 12, "Is there a restocking fee? Most returns have no restocking fee except opened electronics accessories."),
    ("faq_international", "faq", 30, "International returns: customer pays return shipping unless item is defective."),
    ("faq_gift_return", "faq", 8, "Gift returns require the order number from the gift receipt; refunds go as store credit."),
    ("faq_partial_refund", "faq", 22, "Partial refunds apply when only some items in a multi-item order are returned."),
    ("faq_refund_method", "faq", 5, "Refunds return to the original payment method within 5–10 business days after approval."),
    ("faq_price_match", "faq", 40, "Price-match requests must be filed within 14 days of purchase with competitor link."),
    ("faq_missing_parts", "faq", 9, "Missing parts: report within 7 days for a free parts shipment."),
    ("faq_wrong_item", "faq", 3, "Wrong item received: free return label and priority replacement."),
    ("faq_open_box", "faq", 18, "Open-box items have a 14-day return window and may show cosmetic wear."),
    ("faq_loyalty", "faq", 25, "Loyalty points are reversed when an order is fully refunded."),
    ("faq_tax_refund", "faq", 14, "Sales tax is refunded automatically with the merchandise refund."),
    ("faq_hazmat", "faq", 60, "Hazardous materials and lithium batteries cannot be returned by mail."),
    ("faq_custom_orders", "faq", 33, "Custom-engraved products are final sale unless defective."),
    ("faq_bopis", "faq", 11, "Buy-online-pickup-in-store returns must go to the pickup store."),
    ("faq_outlet", "faq", 45, "Outlet purchases are final sale except for manufacturing defects."),
    ("faq_student", "faq", 20, "Student discount orders follow the same refund windows as standard orders."),
    ("faq_preorder", "faq", 7, "Preorder cancellations before ship date are fully refundable."),
    ("faq_bundle", "faq", 16, "Bundle discounts are recalculated if only part of a bundle is returned."),
    ("faq_carrier_delay", "faq", 2, "Carrier delays do not extend the refund window; contact support for exceptions."),
]

_DISTRACTORS = [
    ("ship_rates", "distractor", 4, "Shipping rates: standard 5–7 days, express 2 days, overnight next business day."),
    ("ship_intl", "distractor", 20, "International shipping available to 40 countries; duties paid by recipient."),
    ("acct_password", "distractor", 1, "Reset your password from the login page using your registered email."),
    ("acct_2fa", "distractor", 15, "Enable two-factor authentication under Security settings."),
    ("bill_invoice", "distractor", 6, "Download invoices from Account → Orders → Invoice PDF."),
    ("bill_vat", "distractor", 50, "VAT IDs can be added before checkout for eligible business accounts."),
    ("priv_cookies", "distractor", 90, "Cookie policy: essential cookies only by default; analytics opt-in required."),
    ("priv_gdpr", "distractor", 80, "GDPR data export requests are fulfilled within 30 days via privacy@example.com."),
    ("careers", "distractor", 12, "We are hiring warehouse associates and customer support specialists."),
    ("store_hours", "distractor", 3, "Flagship store hours Monday–Saturday 10am–8pm, Sunday 11am–6pm."),
    ("app_ios", "distractor", 8, "Download our iOS app from the App Store for order tracking."),
    ("app_android", "distractor", 9, "Android app available on Google Play with the same feature set."),
]


def _l2_normalize(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _topic_basis(topic: str, qtext: str) -> Tuple[List[float], List[float]]:
    """Orthonormal-ish (query, orthogonal) pair for controlled cosine injection."""
    q = embed(qtext)
    # Pseudo-orthogonal direction from topic hash — Gram-Schmidt against q
    raw = embed(topic + "::ortho::" + qtext[::-1])
    dot = cosine(raw, q)
    ortho = _l2_normalize([r - dot * qq for r, qq in zip(raw, q)])
    return q, ortho


def _policy_embedding(
    q_basis: Sequence[float],
    ortho: Sequence[float],
    target_cosine: float,
    version: int,
) -> List[float]:
    """
    Build a unit vector with cosine(query) ≈ target_cosine.

    Mirrors the live demo failure mode: stale policies score ~0.83–0.84 while
    the current policy scores ~0.82 — pure cosine prefers outdated text.
    """
    # Mix a tiny version-specific direction so docs are not identical vectors
    jitter = embed(f"version-{version}-jitter")
    j_dot_q = cosine(jitter, q_basis)
    j_dot_o = cosine(jitter, ortho)
    jitter = _l2_normalize(
        [j - j_dot_q * qq - j_dot_o * oo for j, qq, oo in zip(jitter, q_basis, ortho)]
    )
    # v = α·q + √(1-α²)·(β·ortho + γ·jitter), α = target_cosine
    alpha = max(-0.999, min(0.999, target_cosine))
    rest = math.sqrt(max(0.0, 1.0 - alpha * alpha))
    beta, gamma = 0.92, 0.08
    mix = _l2_normalize(
        [beta * o + gamma * j for o, j in zip(ortho, jitter)]
    )
    return _l2_normalize(
        [alpha * qq + rest * m for qq, m in zip(q_basis, mix)]
    )


# Target cosines: oldest stale highest, current lowest-among-policies
# Gap matches README demo (~0.837 stale vs ~0.822 fresh).
_STALE_COSINES = (0.837, 0.832, 0.827)  # v1, v2, v3
_CURRENT_COSINE = 0.822


def build_corpus() -> Tuple[List[Doc], List[Query]]:
    docs: List[Doc] = []
    queries: List[Query] = []

    for topic, qtext, versions in _POLICY_SPECS:
        q_basis, ortho = _topic_basis(topic, qtext)
        stale_ids: List[str] = []
        gold_id = ""
        for ver_i, (age, is_current, body) in enumerate(versions, start=1):
            doc_id = f"{topic}_v{ver_i}"
            if is_current:
                gold_id = doc_id
                target = _CURRENT_COSINE
            else:
                stale_ids.append(doc_id)
                target = _STALE_COSINES[ver_i - 1]
            docs.append(
                Doc(
                    id=doc_id,
                    topic=topic,
                    text=body,
                    age_days=age,
                    is_current=is_current,
                    category="policy",
                    version=ver_i,
                    embedding=_policy_embedding(q_basis, ortho, target, ver_i),
                )
            )
        queries.append(
            Query(
                id=f"q_{topic}",
                text=qtext,
                topic=topic,
                gold_id=gold_id,
                stale_ids=stale_ids,
            )
        )

    for doc_id, cat, age, text in _FAQ_TEXTS + _DISTRACTORS:
        docs.append(
            Doc(
                id=doc_id,
                topic=cat,
                text=text,
                age_days=age,
                is_current=False,
                category=cat,
                embedding=embed(text),
            )
        )

    assert len(docs) >= 50, f"expected ≥50 docs, got {len(docs)}"
    return docs, queries

# ---------------------------------------------------------------------------
# Retrievers (in-process models of each strategy)
# ---------------------------------------------------------------------------

@dataclass
class Hit:
    id: str
    score: float
    semantic: float
    temporal: float
    age_days: int


def search_qdrant_cosine(docs: Sequence[Doc], qvec: Sequence[float], top_k: int) -> List[Hit]:
    scored = [
        Hit(d.id, cosine(qvec, d.embedding), cosine(qvec, d.embedding), 1.0, d.age_days)
        for d in docs
    ]
    scored.sort(key=lambda h: h.score, reverse=True)
    return scored[:top_k]


def search_qdrant_filter(
    docs: Sequence[Doc], qvec: Sequence[float], top_k: int, max_age_days: int
) -> List[Hit]:
    """Qdrant-style: hard metadata filter on timestamp, then cosine rank."""
    filtered = [d for d in docs if d.age_days <= max_age_days]
    if not filtered:
        return []
    return search_qdrant_cosine(filtered, qvec, top_k)


def search_tidevec(
    docs: Sequence[Doc],
    qvec: Sequence[float],
    top_k: int,
    half_life: float,
    beta: float,
) -> List[Hit]:
    hits: List[Hit] = []
    for d in docs:
        sem = cosine(qvec, d.embedding)
        tw = temporal_weight(d.age_days, half_life)
        hits.append(Hit(d.id, blend_score(sem, d.age_days, half_life, beta), sem, tw, d.age_days))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


# Optional live Qdrant validation ------------------------------------------------

def search_live_qdrant_filter(
    docs: Sequence[Doc],
    qvec: Sequence[float],
    top_k: int,
    max_age_days: int,
    url: str,
    collection: str = "tidevec_bench",
) -> List[Hit]:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qm
    except ImportError as e:
        raise SystemExit(
            "qdrant-client not installed. pip install qdrant-client  OR omit --qdrant-url"
        ) from e

    client = QdrantClient(url=url, prefer_grpc=False, check_compatibility=False)
    # Recreate collection with payload index on age_days
    try:
        client.delete_collection(collection)
    except Exception:
        pass
    client.create_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=len(docs[0].embedding), distance=qm.Distance.COSINE),
    )
    client.create_payload_index(collection, "age_days", field_schema=qm.PayloadSchemaType.INTEGER)
    points = [
        qm.PointStruct(
            id=i,
            vector=d.embedding,
            payload={"doc_id": d.id, "age_days": d.age_days},
        )
        for i, d in enumerate(docs)
    ]
    client.upsert(collection_name=collection, points=points)
    res = client.search(
        collection_name=collection,
        query_vector=list(qvec),
        limit=top_k,
        query_filter=qm.Filter(
            must=[qm.FieldCondition(key="age_days", range=qm.Range(lte=max_age_days))]
        ),
    )
    return [
        Hit(r.payload["doc_id"], float(r.score), float(r.score), 1.0, int(r.payload["age_days"]))
        for r in res
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rank_of(hits: Sequence[Hit], doc_id: str) -> Optional[int]:
    for i, h in enumerate(hits, start=1):
        if h.id == doc_id:
            return i
    return None


def ndcg_at_k(hits: Sequence[Hit], relevance: Dict[str, float], k: int = 5) -> float:
    def dcg(rels: List[float]) -> float:
        return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))

    gained = [relevance.get(h.id, 0.0) for h in hits[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return (dcg(gained) / idcg) if idcg > 0 else 0.0


def relevance_map(q: Query, docs: Sequence[Doc]) -> Dict[str, float]:
    """Graded relevance: current=3, other same-topic versions=1, else 0.
    Stale same-topic still gets 1 (topically relevant) but gold gets 3 —
    NDCG penalises putting stale above current."""
    rel: Dict[str, float] = {}
    for d in docs:
        if d.id == q.gold_id:
            rel[d.id] = 3.0
        elif d.topic == q.topic and d.category == "policy":
            rel[d.id] = 1.0
        else:
            rel[d.id] = 0.0
    return rel


@dataclass
class StrategyStats:
    name: str
    gold_at_1: float
    stale_at_1: float
    mean_gold_rank: float
    median_gold_rank: float
    ndcg_at_5: float
    empty_result_rate: float
    per_query: List[dict] = field(default_factory=list)


def evaluate(
    name: str,
    retriever: Callable[[Query], List[Hit]],
    queries: Sequence[Query],
    docs: Sequence[Doc],
    top_k: int = 10,
) -> StrategyStats:
    gold_at_1 = 0
    stale_at_1 = 0
    empty = 0
    gold_ranks: List[float] = []
    ndcgs: List[float] = []
    per_query: List[dict] = []

    doc_by_id = {d.id: d for d in docs}

    for q in queries:
        hits = retriever(q)
        if not hits:
            empty += 1
            gold_ranks.append(float(top_k + 1))  # miss
            ndcgs.append(0.0)
            per_query.append({"query": q.id, "top1": None, "gold_rank": None, "empty": True})
            continue

        top1 = hits[0].id
        if top1 == q.gold_id:
            gold_at_1 += 1
        if top1 in q.stale_ids:
            stale_at_1 += 1

        gr = rank_of(hits, q.gold_id)
        gold_ranks.append(float(gr if gr is not None else top_k + 1))
        nd = ndcg_at_k(hits, relevance_map(q, docs), k=5)
        ndcgs.append(nd)

        per_query.append(
            {
                "query": q.id,
                "text": q.text,
                "top1": top1,
                "top1_age_days": doc_by_id[top1].age_days if top1 in doc_by_id else None,
                "top1_is_gold": top1 == q.gold_id,
                "top1_is_stale": top1 in q.stale_ids,
                "gold_rank": gr,
                "ndcg@5": round(nd, 4),
                "top5": [
                    {
                        "id": h.id,
                        "score": round(h.score, 4),
                        "semantic": round(h.semantic, 4),
                        "temporal": round(h.temporal, 4),
                        "age_days": h.age_days,
                    }
                    for h in hits[:5]
                ],
            }
        )

    n = len(queries)
    return StrategyStats(
        name=name,
        gold_at_1=gold_at_1 / n,
        stale_at_1=stale_at_1 / n,
        mean_gold_rank=statistics.mean(gold_ranks),
        median_gold_rank=statistics.median(gold_ranks),
        ndcg_at_5=statistics.mean(ndcgs),
        empty_result_rate=empty / n,
        per_query=per_query,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_table(stats: Sequence[StrategyStats]) -> None:
    headers = [
        "Strategy",
        "Gold@1 ↑",
        "Stale@1 ↓",
        "Mean gold rank ↓",
        "nDCG@5 ↑",
        "Empty ↓",
    ]
    rows = [
        [
            s.name,
            f"{s.gold_at_1*100:5.1f}%",
            f"{s.stale_at_1*100:5.1f}%",
            f"{s.mean_gold_rank:5.2f}",
            f"{s.ndcg_at_5:5.3f}",
            f"{s.empty_result_rate*100:5.1f}%",
        ]
        for s in stats
    ]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:{w}}}" for w in widths)
    print()
    print(fmt.format(*headers))
    print(fmt.format(*["─" * w for w in widths]))
    for r in rows:
        print(fmt.format(*r))
    print()


# Topics whose current policy is >14 days old (hard-filter trap cohort)
_STALE_CURRENT_TOPICS = {
    "apparel_exchange",
    "warranty_claim",
    "subscription_cancel",
    "damaged_item",
}


def print_cohort_breakdown(stats: Sequence[StrategyStats], queries: Sequence[Query]) -> None:
    """Show why filter≤14d looks good in aggregate but fails on authoritative-but-old docs."""
    fresh_q = [q for q in queries if q.topic not in _STALE_CURRENT_TOPICS]
    old_q = [q for q in queries if q.topic in _STALE_CURRENT_TOPICS]
    print("Cohort breakdown (Gold@1)")
    print("─────────────────────────")
    print(f"  {'Strategy':28s}  {'Fresh current (n='+str(len(fresh_q))+')':22s}  {'Current >14d (n='+str(len(old_q))+')'}")
    for s in stats:
        by_q = {p["query"]: p for p in s.per_query}
        def rate(qs: Sequence[Query]) -> str:
            if not qs:
                return "  n/a"
            hits = sum(1 for q in qs if by_q[q.id].get("top1_is_gold"))
            return f"{100*hits/len(qs):5.1f}%"
        print(f"  {s.name:28s}  {rate(fresh_q):22s}  {rate(old_q)}")
    print()



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--half-life-days", type=float, default=7.0)
    ap.add_argument("--blend", type=float, default=0.3, help="TideVec temporal_blend β")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--filter-14", type=int, default=14, help="Hard filter window (days)")
    ap.add_argument("--filter-30", type=int, default=30, help="Looser hard filter window")
    ap.add_argument("--json", type=Path, default=Path("results/benchmark_results.json"))
    ap.add_argument("--qdrant-url", default=None, help="Optional live Qdrant URL for filter validation")
    args = ap.parse_args()

    docs, queries = build_corpus()
    print(f"Corpus: {len(docs)} documents, {len(queries)} queries")
    print(f"Policy docs: {sum(1 for d in docs if d.category=='policy')}")
    print(f"TideVec config: half_life={args.half_life_days}d, blend={args.blend}")

    strategies: List[Tuple[str, Callable[[Query], List[Hit]]]] = [
        (
            "Qdrant cosine-only",
            lambda q, _d=docs: search_qdrant_cosine(_d, embed(q.text), args.top_k),
        ),
        (
            f"Qdrant filter ≤{args.filter_14}d",
            lambda q, _d=docs: search_qdrant_filter(_d, embed(q.text), args.top_k, args.filter_14),
        ),
        (
            f"Qdrant filter ≤{args.filter_30}d",
            lambda q, _d=docs: search_qdrant_filter(_d, embed(q.text), args.top_k, args.filter_30),
        ),
        (
            f"TideVec blend β={args.blend}",
            lambda q, _d=docs: search_tidevec(
                _d, embed(q.text), args.top_k, args.half_life_days, args.blend
            ),
        ),
    ]

    if args.qdrant_url:
        strategies.append(
            (
                f"Live Qdrant filter ≤{args.filter_14}d",
                lambda q, _d=docs: search_live_qdrant_filter(
                    _d, embed(q.text), args.top_k, args.filter_14, args.qdrant_url
                ),
            )
        )

    all_stats = [evaluate(name, fn, queries, docs, args.top_k) for name, fn in strategies]
    print_table(all_stats)
    print_cohort_breakdown(all_stats, queries)

    cosine_s = all_stats[0]
    filter14_s = all_stats[1]
    filter30_s = all_stats[2]
    tide_s = next(s for s in all_stats if s.name.startswith("TideVec"))
    print("Key takeaways")
    print("─────────────")
    print(
        f"· Cosine-only returns a STALE policy at rank 1 on "
        f"{cosine_s.stale_at_1*100:.0f}% of queries — same failure mode as the live demo "
        f"(stale semantic 0.837 > current 0.822)."
    )
    print(
        f"· Hard ≤{args.filter_14}d filter: overall Gold@1={filter14_s.gold_at_1*100:.0f}%, "
        f"but 0% on the cohort where the authoritative policy is itself >14 days old "
        f"(excluded by the metadata filter)."
    )
    print(
        f"· Hard ≤{args.filter_30}d filter: Gold@1={filter30_s.gold_at_1*100:.0f}%, "
        f"Stale@1={filter30_s.stale_at_1*100:.0f}% — widening the window re-admits "
        f"outdated versions that still beat current on cosine."
    )
    print(
        f"· TideVec soft decay: Gold@1={tide_s.gold_at_1*100:.0f}%, "
        f"Stale@1={tide_s.stale_at_1*100:.0f}%, nDCG@5={tide_s.ndcg_at_5:.3f} — "
        f"demotes stale continuously; no window tuning required."
    )
    print(
        "· Bottom line: a timestamp filter is binary include/exclude. TideVec is a "
        "continuous rerank — stale docs stay searchable but lose beam priority."
    )

    out = {
        "config": {
            "half_life_days": args.half_life_days,
            "temporal_blend": args.blend,
            "top_k": args.top_k,
            "n_docs": len(docs),
            "n_queries": len(queries),
            "embedding": f"controlled_cosine_dim_{DIM}",
            "stale_semantic": _STALE_COSINES[0],
            "current_semantic": _CURRENT_COSINE,
            "note": (
                "Policy embeddings use controlled cosines matching the live demo "
                "(stale 0.837 > current 0.822). FAQ/distractor docs use hashed BoW."
            ),
        },
        "strategies": [
            {
                "name": s.name,
                "gold_at_1": round(s.gold_at_1, 4),
                "stale_at_1": round(s.stale_at_1, 4),
                "mean_gold_rank": round(s.mean_gold_rank, 4),
                "median_gold_rank": round(s.median_gold_rank, 4),
                "ndcg_at_5": round(s.ndcg_at_5, 4),
                "empty_result_rate": round(s.empty_result_rate, 4),
                "gold_at_1_fresh_current": round(
                    sum(
                        1
                        for p in s.per_query
                        if p["query"].replace("q_", "") not in _STALE_CURRENT_TOPICS
                        and p.get("top1_is_gold")
                    )
                    / max(
                        1,
                        sum(
                            1
                            for p in s.per_query
                            if p["query"].replace("q_", "") not in _STALE_CURRENT_TOPICS
                        ),
                    ),
                    4,
                ),
                "gold_at_1_current_gt_14d": round(
                    sum(
                        1
                        for p in s.per_query
                        if p["query"].replace("q_", "") in _STALE_CURRENT_TOPICS
                        and p.get("top1_is_gold")
                    )
                    / max(
                        1,
                        sum(
                            1
                            for p in s.per_query
                            if p["query"].replace("q_", "") in _STALE_CURRENT_TOPICS
                        ),
                    ),
                    4,
                ),
                "per_query": s.per_query,
            }
            for s in all_stats
        ],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
