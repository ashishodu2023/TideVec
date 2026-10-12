import os
import random
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _server_binary():
    candidate = os.environ.get("TIDEVEC_SERVER_BIN") or str(REPO / "build" / "tidevec-server")
    return candidate if os.path.exists(candidate) else shutil.which("tidevec-server")


def _free_ports(n: int) -> list:
    """
    n distinct free ports below the Linux ephemeral range (32768+), so
    outgoing connections between running servers can't grab a port a
    server is about to bind.
    """
    rng = random.Random()
    ports: list = []
    while len(ports) < n:
        port = rng.randint(20000, 32000)
        if port in ports:
            continue
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
        ports.append(port)
    return ports


def _free_port() -> int:
    return _free_ports(1)[0]


@pytest.fixture(scope="session")
def server_url(tmp_path_factory):
    binary = _server_binary()
    if not binary:
        pytest.skip("tidevec-server binary not found (build it or set TIDEVEC_SERVER_BIN)")
    port = _free_port()
    data = tmp_path_factory.mktemp("server-data")
    proc = subprocess.Popen(
        [binary, "--port", str(port), "--data-dir", str(data), "--no-require-auth", "--quiet"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    from tidevec import TideVec
    client = TideVec(f"127.0.0.1:{port}", timeout=5)
    for _ in range(100):
        if client.ping():
            break
        if proc.poll() is not None:
            pytest.fail(f"tidevec-server exited with code {proc.returncode}")
        time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("tidevec-server did not start")
    yield f"127.0.0.1:{port}"
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def local_db(tmp_path):
    pytest.importorskip("tidevec_native")
    from tidevec import Local
    db = Local(str(tmp_path / "db"))
    yield db
    db.close()


@pytest.fixture(params=["local", "server"])
def db(request, tmp_path):
    """The same tests run against embedded mode and a live server."""
    if request.param == "local":
        yield request.getfixturevalue("local_db")
    else:
        from tidevec import TideVec
        client = TideVec(request.getfixturevalue("server_url"))
        created = set(c.name for c in client.list_collections())
        yield client
        for c in client.list_collections():
            if c.name not in created:
                client.drop_collection(c.name)


@pytest.fixture
def name(request):
    return "t_" + request.node.name.replace("[", "_").replace("]", "").replace("-", "_")[:40]
