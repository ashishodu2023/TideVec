"""DriftBridge model migration against a live server."""

import hashlib
import json
import math
import random
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tidevec import APIError, ForbiddenError, TideVec

from conftest import _free_port, _server_binary

DIM = 8


def model(seed: int, text: str):
    """Deterministic stand-in for an embedding model version."""
    rng = random.Random(hashlib.sha256(f"{seed}|{text}".encode()).digest())
    vec = [rng.gauss(0.0, 1.0) for _ in range(DIM)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


@pytest.fixture
def reembed_url():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            time.sleep(0.01)
            out = json.dumps({"embedding": model(2, body["payload"]["text"])}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/reembed"
    httpd.shutdown()


@pytest.fixture
def private_server(tmp_path):
    binary = _server_binary()
    if not binary:
        pytest.skip("tidevec-server binary not found")
    port = _free_port()
    proc = subprocess.Popen(
        [binary, "--port", str(port), "--data-dir", str(tmp_path), "--no-require-auth", "--quiet",
         "--rate-limit", "100000", "--reembed-allow", "127.0.0.1", "--reembed-allow-private"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    client = TideVec(f"127.0.0.1:{port}", timeout=5)
    for _ in range(100):
        if client.ping():
            break
        time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("tidevec-server did not start")
    yield client
    proc.terminate()
    proc.wait(timeout=10)


def _wait_done(db, name, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = db.drift_status(name)
        if st.phase in ("COMPLETE", "FAILED"):
            return st
        time.sleep(0.1)
    pytest.fail("migration did not finish")


def test_migration_respects_concurrent_writes(private_server, reembed_url):
    db = private_server
    texts = {f"d{i}": f"document number {i}" for i in range(60)}
    db.create_collection("kb", dim=DIM, n_shards=2)
    db.upsert("kb", [{"id": i, "embedding": model(1, t), "payload": {"text": t}} for i, t in texts.items()])

    db.start_drift("kb", reembed_url)
    assert db.drift_status("kb").phase == "MIGRATING"
    db.delete("kb", ["d59"])
    texts["d0"] = "document zero, rewritten during the migration"
    db.upsert("kb", [{"id": "d0", "embedding": model(1, texts["d0"]), "payload": {"text": texts["d0"]}}])
    del texts["d59"]

    st = _wait_done(db, "kb")
    assert st.phase == "COMPLETE"
    assert db.count("kb") == 59
    for i, t in texts.items():
        hit = db.search("kb", model(2, t), top_k=1, temporal_blend=0.0)[0]
        assert hit.id == i and hit.payload["text"] == t
        assert hit.vector_score == pytest.approx(1.0, abs=1e-4)


def test_migration_rejects_dimension_change(private_server, reembed_url):
    private_server.create_collection("kb2", dim=DIM)
    with pytest.raises(APIError, match="new_dim"):
        private_server.start_drift("kb2", reembed_url, new_dim=DIM * 2)


def test_private_reembed_url_needs_opt_in(server_url, reembed_url):
    db = TideVec(server_url)
    db.create_collection("drift_ssrf", dim=DIM)
    try:
        with pytest.raises(ForbiddenError):
            db.start_drift("drift_ssrf", reembed_url)
    finally:
        db.drop_collection("drift_ssrf")
