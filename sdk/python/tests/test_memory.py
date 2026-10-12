import asyncio
import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from tidevec.embeddings import HashingEmbedder
from tidevec.memory import AgentMemory


@pytest.fixture
def memory(db, name):
    return AgentMemory(db, name, HashingEmbedder(256))


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder(64)
    a, b = e.embed_query("Refunds take 14 days"), e.embed_query("refunds take 14 days")
    assert a == b
    assert sum(x * x for x in a) == pytest.approx(1.0)
    assert e.embed_documents(["x", "y"])[0] == e.embed_query("x")


def test_keyed_memory_supersedes_previous_value(memory):
    first = memory.remember("The user's favourite editor is Vim", key="user.editor")
    second = memory.remember("The user switched editors and now prefers Helix", key="user.editor")
    assert second["superseded"] == [first["id"]]

    current = memory.recall("which editor does the user like?", k=3)
    texts = [m["text"] for m in current]
    assert "The user switched editors and now prefers Helix" in texts
    assert "The user's favourite editor is Vim" not in texts

    everything = memory.recall("which editor does the user like?", k=3, include_superseded=True)
    old = next(m for m in everything if m["id"] == first["id"])
    assert old["superseded_by"] == [second["id"]]

    versions = memory.history("user.editor")
    assert [v["id"] for v in versions] == [first["id"], second["id"]]


def test_third_version_only_supersedes_current_head(memory):
    a = memory.remember("timezone is UTC", key="user.tz")
    b = memory.remember("timezone is CET", key="user.tz")
    c = memory.remember("timezone is PST", key="user.tz")
    assert b["superseded"] == [a["id"]]
    assert c["superseded"] == [b["id"]]
    assert [m["id"] for m in memory.recall("timezone", k=5)] == [c["id"]]


def test_forget_by_key_and_id(memory):
    a = memory.remember("project uses Postgres", key="project.db")
    memory.remember("project moved to SQLite", key="project.db")
    keep = memory.remember("deploys happen on Fridays")
    assert memory.forget(key="project.db") == 2
    assert memory.forget(memory_id=keep["id"]) == 1
    assert memory.recall("database", k=5) == []
    assert memory.forget(memory_id=a["id"]) == 0


_DAY = 86_400_000


def test_recall_and_history_as_of(memory):
    now = int(time.time() * 1000)
    v1 = memory.remember("The team standup is at 9am", key="team.standup",
                         created_at=now - 10 * _DAY)
    v2 = memory.remember("The team standup moved to 10am", key="team.standup",
                         created_at=now - 3 * _DAY)
    other = memory.remember("Lunch is at noon", created_at=now - 1 * _DAY)

    current = memory.recall("when is the standup", k=5)
    assert v2["id"] in [m["id"] for m in current]
    assert v1["id"] not in [m["id"] for m in current]

    week_ago = datetime.fromtimestamp((now - 7 * _DAY) / 1000, timezone.utc)
    for as_of in (week_ago, week_ago.isoformat(), now - 7 * _DAY,
                  (week_ago - timedelta(days=1)).date()):
        then = memory.recall("when is the standup", k=5, as_of=as_of)
        assert [m["id"] for m in then] == [v1["id"]], as_of
        assert then[0]["superseded_by"] == []
    assert memory.recall("standup", k=5, as_of=now - 20 * _DAY) == []
    assert other["id"] in [m["id"] for m in memory.recall("lunch", k=5)]

    full = memory.history("team.standup")
    assert [v["id"] for v in full] == [v1["id"], v2["id"]]
    assert full[0]["superseded_by"] == [v2["id"]] and full[1]["superseded_by"] == []
    assert full[0]["text"] == "The team standup is at 9am" and full[0]["key"] == "team.standup"

    past = memory.history("team.standup", as_of=week_ago)
    assert [v["id"] for v in past] == [v1["id"]] and past[0]["superseded_by"] == []
    assert memory.history("team.standup", as_of=now - 20 * _DAY) == []
    with pytest.raises(ValueError):
        memory.history("team.standup", as_of="last tuesday")


def test_recall_metadata_filter_and_get(memory):
    a = memory.remember("Alpha ships in March", metadata={"project": "alpha", "priority": 3})
    b = memory.remember("Alpha budget is 40k", metadata={"project": "alpha", "priority": 1})
    c = memory.remember("Beta ships in May", metadata={"project": "beta", "priority": 2})

    ids = lambda ms: {m["id"] for m in ms}  # noqa: E731
    assert ids(memory.recall("ships", k=5, filter={"project": "alpha"})) == {a["id"], b["id"]}
    assert ids(memory.recall("ships", k=5, filter={"priority": {"$gte": 2}})) == {a["id"], c["id"]}
    assert ids(memory.recall("ships", k=5, filter={"$or": [{"project": "beta"},
                                                           {"priority": 1}]})) == {b["id"], c["id"]}
    keyed = memory.remember("Alpha ships in April", key="alpha.ship",
                            metadata={"project": "alpha"})
    assert ids(memory.recall("ships", k=5, key="alpha.ship",
                             filter={"project": "alpha"})) == {keyed["id"]}
    assert memory.recall("ships", k=5, key="alpha.ship", filter={"project": "beta"}) == []

    got = memory.get(a["id"])
    assert got["text"] == "Alpha ships in March" and got["metadata"]["project"] == "alpha"
    assert got["superseded_by"] == [] and got["key"] is None
    assert memory.get("missing") is None


def test_history_finds_every_version(memory):
    start = int(time.time() * 1000) - 60_000
    ids = [memory.remember(f"counter value is {i}", key="counter", created_at=start + i)["id"]
           for i in range(30)]
    versions = memory.history("counter")
    assert [v["id"] for v in versions] == ids
    assert all(v["superseded_by"] for v in versions[:-1]) and not versions[-1]["superseded_by"]
    assert memory.forget(key="counter") == 30


def test_contradiction_judge_records_edges(db, name):
    def judge(new, old):
        return "contradicts" if "closed" in new and "open" in old else "unrelated"

    mem = AgentMemory(db, name, HashingEmbedder(256), contradiction_judge=judge)
    old = mem.remember("The office is open on Saturdays")
    new = mem.remember("The office is closed on Saturdays")
    assert new["relations"] == [{"tgt": old["id"], "type": "CONTRADICTS"}]
    hits = db.search(name, HashingEmbedder(256).embed_query("office Saturdays"),
                     top_k=2, mode="contradiction_check")
    target = next(h for h in hits if h.id == old["id"])
    assert new["id"] in target.contradicted_by


def _tool_payload(result):
    if isinstance(result, tuple):  # mcp 1.x returns (content, structured)
        result = result[1] if result[1] is not None else result[0]
    if isinstance(result, dict):
        return result.get("result", result)
    structured = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
    if structured is not None:
        return structured.get("result", structured)
    content = result.content if hasattr(result, "content") else result
    return json.loads(content[0].text)


def test_mcp_server_tools(local_db):
    pytest.importorskip("mcp")
    from tidevec.mcp_server import build_server

    server = build_server(AgentMemory(local_db, "mcp_mem", HashingEmbedder(128)))
    tools = {t.name for t in asyncio.run(server.list_tools())}
    assert tools == {"remember", "recall", "history", "forget"}

    first = _tool_payload(asyncio.run(server.call_tool(
        "remember", {"text": "deadline is March 3", "key": "project.deadline"})))
    second = _tool_payload(asyncio.run(server.call_tool(
        "remember", {"text": "deadline moved to April 9", "key": "project.deadline"})))
    assert second["superseded"] == [first["id"]]

    recalled = _tool_payload(asyncio.run(server.call_tool("recall", {"query": "deadline"})))
    assert [m["text"] for m in recalled] == ["deadline moved to April 9"]

    # as_of / filter: what was known before the update, scoped by metadata.
    mem = AgentMemory(local_db, "mcp_mem_t", HashingEmbedder(128))
    now = int(time.time() * 1000)
    old = mem.remember("release is v1", key="release", metadata={"team": "core"},
                       created_at=now - 5 * _DAY)
    mem.remember("release is v2", key="release", metadata={"team": "core"},
                 created_at=now - 1 * _DAY)
    mem.remember("release notes live in the wiki", metadata={"team": "docs"})
    server = build_server(mem)
    before = datetime.fromtimestamp((now - 3 * _DAY) / 1000, timezone.utc).isoformat()
    then = _tool_payload(asyncio.run(server.call_tool(
        "recall", {"query": "release", "as_of": before, "filter": {"team": "core"}})))
    assert [m["id"] for m in then] == [old["id"]]
    docs = _tool_payload(asyncio.run(server.call_tool(
        "recall", {"query": "release", "filter": {"team": "docs"}})))
    assert [m["text"] for m in docs] == ["release notes live in the wiki"]
    hist = _tool_payload(asyncio.run(server.call_tool(
        "history", {"key": "release", "as_of": before})))
    assert [m["id"] for m in hist] == [old["id"]] and hist[0]["superseded_by"] == []
