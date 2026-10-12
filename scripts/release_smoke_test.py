#!/usr/bin/env python3
"""Release smoke test: run the README quick-start against a server or embedded mode.

    python scripts/release_smoke_test.py --server localhost:6399 [--expect-version 0.3.0]
    python scripts/release_smoke_test.py --embedded            [--expect-version 0.3.0]

Needs the `tidevec` Python package (and `tidevec-embedded` for --embedded).
For --server, the API key is read from TIDEVEC_API_KEY. Exits non-zero on failure.
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
import urllib.request

DAY = 86_400_000


def wait_for_health(host: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    last_err = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://{host}/health", timeout=2) as r:
                return json.loads(r.read())
        except Exception as e:  # server still starting
            last_err = e
            time.sleep(0.5)
    raise SystemExit(f"server at {host} not healthy after {timeout_s}s: {last_err}")


def run_quickstart(db, version_reported: str, expect_version: str) -> None:
    from tidevec import HalfLife
    from tidevec.embeddings import HashingEmbedder

    if expect_version and version_reported != expect_version:
        raise SystemExit(f"version mismatch: reported {version_reported!r}, "
                         f"expected {expect_version!r}")

    emb = HashingEmbedder(256)
    now = int(time.time() * 1000)
    name = f"smoke_{int(time.time())}"
    db.create_collection(name, dim=emb.dim, half_life_ms=HalfLife.ONE_MONTH,
                         temporal_blend=0.3, supersession=True,
                         supersede_match_field="doc_id", supersede_threshold=0.5)
    try:
        old = "Refund policy: refunds are issued within 14 days of the return request."
        new = "Refund policy update: refunds are now issued within 3 days."
        db.upsert(name, [{"id": "refunds-2025", "embedding": emb.embed_query(old),
                          "created_at": now - 200 * DAY,
                          "payload": {"doc_id": "refunds", "text": old}}])
        db.upsert(name, [{"id": "refunds-2026", "embedding": emb.embed_query(new),
                          "created_at": now - 2 * DAY,
                          "payload": {"doc_id": "refunds", "text": new}}])

        query = emb.embed_query("How many days after the return request are refunds issued?")
        hits = list(db.search(name, query, top_k=5, superseded_penalty=1.0))
        ids = [h.id for h in hits]
        print("hits:", [(h.id, round(h.score, 3), h.superseded_by) for h in hits])

        assert ids and ids[0] == "refunds-2026", f"current version not first: {ids}"
        old_hit = next((h for h in hits if h.id == "refunds-2025"), None)
        assert old_hit is not None, f"old version missing from results: {ids}"
        assert old_hit.superseded_by == ["refunds-2026"], \
            f"supersession not linked: {old_hit.superseded_by}"

        try:
            db.upsert(name, [{"id": "bad-dim", "embedding": [0.1, 0.2]}])
        except Exception:
            pass
        else:
            raise AssertionError("upsert with the wrong dimension was accepted")
    finally:
        db.drop_collection(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--server", metavar="HOST:PORT")
    mode.add_argument("--embedded", action="store_true")
    ap.add_argument("--expect-version", default="")
    ap.add_argument("--wait", type=float, default=60.0, help="seconds to wait for /health")
    args = ap.parse_args()

    import tidevec
    print(f"tidevec Python package {tidevec.__version__}")
    if args.expect_version and tidevec.__version__ != args.expect_version:
        raise SystemExit(f"tidevec.__version__ is {tidevec.__version__!r}, "
                         f"expected {args.expect_version!r}")

    if args.server:
        health = wait_for_health(args.server, args.wait)
        print("health:", health)
        db = tidevec.TideVec(args.server)
        run_quickstart(db, health.get("version", ""), args.expect_version)
    else:
        from importlib.metadata import version
        engine_version = version("tidevec-embedded")
        print(f"tidevec-embedded {engine_version}")
        path = tempfile.mkdtemp(prefix="tidevec-smoke-")
        try:
            db = tidevec.Local(path)
            run_quickstart(db, engine_version, args.expect_version)
            db.close()
        finally:
            shutil.rmtree(path, ignore_errors=True)

    print("smoke test OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
