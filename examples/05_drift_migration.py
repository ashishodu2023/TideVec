"""
Zero-downtime embedding-model migration with DriftBridge.

    python examples/05_drift_migration.py                 # spawns ./build/tidevec-server
    python examples/05_drift_migration.py --url host:6399 # use a running server

Scenario: a collection was embedded with "model v1" and you switch to
"model v2". Queries embedded with v2 no longer match the stored v1 vectors,
and re-ingesting everything means an empty or half-migrated index for a while.

DriftBridge re-embeds every stored vector in the background by calling your
re-embed endpoint (POST {"id", "payload"} -> {"embedding"}), keeps serving
the old vectors until the shadow index is complete, and mirrors writes made
during the migration. Vectors deleted or rewritten mid-migration are not
resurrected or rolled back by the swap.

Both "models" here are offline hashing embedders with different seeds, so
v1 and v2 vectors are unrelated, as they would be across real model versions.
The re-embed service runs on 127.0.0.1, which the server only accepts when
started with:  --reembed-allow 127.0.0.1 --reembed-allow-private
DriftBridge is standalone-only (not replicated in cluster mode) and keeps
the collection dimension; to change dimension, create a new collection.
"""

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from tidevec import TideVec

DIM = 128
COLLECTION = "kb"
REPO = Path(__file__).resolve().parents[1]


def hashing_model(seed: str):
    def embed(text: str):
        vec = [0.0] * DIM
        words = re.findall(r"\w+", text.lower())
        for feat in words + [f"{a} {b}" for a, b in zip(words, words[1:])]:
            h = int.from_bytes(hashlib.blake2b(f"{seed}|{feat}".encode(), digest_size=8).digest(), "little")
            vec[h % DIM] += 1.0 if (h >> 63) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]
    return embed


model_v1 = hashing_model("v1")
model_v2 = hashing_model("v2")

TOPICS = ["refunds", "shipping", "warranty", "billing", "accounts", "security",
          "returns", "pricing", "support", "privacy"]
DOCS = {
    f"doc-{i:03d}": f"{TOPICS[i % len(TOPICS)]} article {i}: how {TOPICS[i % len(TOPICS)]} "
                    f"works for plan {i // len(TOPICS)} customers in region {i % 7}"
    for i in range(300)
}


def start_reembed_service(delay_s: float):
    """The endpoint DriftBridge calls for every stored vector."""
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            time.sleep(delay_s)  # pretend to be a remote embedding API
            out = json.dumps({"embedding": model_v2(body["payload"]["text"])}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/reembed"


def spawn_server(binary: str):
    data = tempfile.mkdtemp(prefix="tidevec-drift-")
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [binary, "--port", str(port), "--data-dir", data, "--no-require-auth", "--quiet",
         "--reembed-allow", "127.0.0.1", "--reembed-allow-private"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    client = TideVec(f"127.0.0.1:{port}")
    for _ in range(100):
        if client.ping():
            return proc, data, client
        time.sleep(0.1)
    proc.kill()
    raise SystemExit("tidevec-server did not start")


def top1(db, query, model):
    hits = db.search(COLLECTION, model(query), top_k=1, temporal_blend=0.0)
    return (hits[0].id, round(hits[0].vector_score, 3)) if hits else (None, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="existing server (must allow the re-embed host)")
    ap.add_argument("--server-bin", default=str(REPO / "build" / "tidevec-server"))
    ap.add_argument("--delay", type=float, default=0.005, help="seconds per re-embed call")
    args = ap.parse_args()

    proc = data = None
    if args.url:
        db = TideVec(args.url)
    else:
        proc, data, db = spawn_server(args.server_bin)
    httpd, reembed_url = start_reembed_service(args.delay)

    try:
        if db.has_collection(COLLECTION):
            db.drop_collection(COLLECTION)
        db.create_collection(COLLECTION, dim=DIM, n_shards=2)
        db.upsert(COLLECTION, [{"id": i, "embedding": model_v1(t), "payload": {"text": t}}
                               for i, t in DOCS.items()])
        query, target = DOCS["doc-042"], "doc-042"
        print(f"{db.count(COLLECTION)} docs embedded with model v1")
        print(f"  v1 query -> {top1(db, query, model_v1)}")
        print(f"  v2 query -> {top1(db, query, model_v2)}   (wrong space: v2 query vs v1 vectors)\n")

        db.start_drift(COLLECTION, reembed_url)
        print("migration started; the live index keeps serving v1 queries")

        # Writes during the migration: one delete, one rewrite, one new doc.
        db.delete(COLLECTION, ["doc-299"])
        rewritten = "refunds article 10: refunds now take 30 days"
        db.upsert(COLLECTION, [
            {"id": "doc-010", "embedding": model_v1(rewritten), "payload": {"text": rewritten}},
            {"id": "doc-new", "embedding": model_v1("brand new pricing page"),
             "payload": {"text": "brand new pricing page"}},
        ])

        while True:
            st = db.drift_status(COLLECTION)
            hit = top1(db, query, model_v1)
            print(f"  {st.phase:<9} {st.pct_complete:5.1f}%   v1 query still -> {hit}")
            if st.phase in ("COMPLETE", "FAILED", "IDLE"):
                break
            time.sleep(0.3)
        if st.phase != "COMPLETE":
            raise SystemExit(f"migration ended in {st.phase}: {st.error}")

        print("\nafter the swap:")
        print(f"  v2 query -> {top1(db, query, model_v2)}   (expected {target})")
        print(f"  count = {db.count(COLLECTION)}  (300 - 1 deleted + 1 new)")
        rw = db.search(COLLECTION, model_v2(rewritten), top_k=1, temporal_blend=0.0)[0]
        print(f"  rewritten doc kept its new text: {rw.id} -> {rw.payload['text']!r}")
        gone = db.search(COLLECTION, model_v2(DOCS["doc-299"]), top_k=1, temporal_blend=0.0)[0]
        print(f"  deleted doc stayed deleted: top hit for its text is {gone.id}")
        assert top1(db, query, model_v2)[0] == target
        assert db.count(COLLECTION) == 300
        assert rw.id == "doc-010" and gone.id != "doc-299"
    finally:
        httpd.shutdown()
        if proc:
            proc.terminate()
            proc.wait(timeout=10)
            shutil.rmtree(data, ignore_errors=True)


if __name__ == "__main__":
    main()
