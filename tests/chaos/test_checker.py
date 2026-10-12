"""Unit tests for the linearizability checker (no server needed)."""
import random

from linearizability import INF, Op, check_history, check_key


def W(c, v, call, ret, status="ok", key="k"):
    return Op(c, "write", key, v, call, ret, status)


def R(c, v, call, ret, status="ok", key="k"):
    return Op(c, "read", key, v, call, ret, status)


def test_sequential_history_is_linearizable():
    h = [R(0, None, 0, 1), W(0, "a", 2, 3), R(1, "a", 4, 5), W(1, "b", 6, 7), R(0, "b", 8, 9)]
    assert check_key("k", h).ok


def test_concurrent_write_allows_either_read():
    h = [W(0, "a", 0, 1), W(1, "b", 2, 10), R(2, "a", 3, 4), R(3, "b", 5, 6)]
    assert check_key("k", h).ok


def test_stale_read_after_completed_write_is_a_violation():
    h = [W(0, "a", 0, 1), W(1, "b", 2, 3), R(2, "a", 4, 5)]
    r = check_key("k", h)
    assert not r.ok and "not linearizable" in r.reason and r.context


def test_reads_cannot_go_back_in_time():
    # b is concurrent with both reads, but once a read sees b a later read can't see a.
    h = [W(0, "a", 0, 1), W(1, "b", 2, 20), R(2, "b", 3, 4), R(3, "a", 5, 6)]
    assert not check_key("k", h).ok


def test_read_of_never_written_value():
    r = check_key("k", [W(0, "a", 0, 1), R(1, "zzz", 2, 3)])
    assert not r.ok and "never written" in r.reason


def test_info_write_may_take_effect_late_or_never():
    # The timed-out write of b shows up long after its call...
    h = [W(0, "a", 0, 1), W(1, "b", 2, INF, "info"), R(2, "a", 3, 4), R(3, "b", 50, 51)]
    assert check_key("k", h).ok
    # ...or never (unobserved info writes are dropped).
    h = [W(0, "a", 0, 1), W(1, "b", 2, INF, "info"), R(2, "a", 3, 4), R(3, "a", 50, 51)]
    assert check_key("k", h).ok


def test_observed_info_write_cannot_be_unseen():
    h = [W(0, "a", 0, 1), W(1, "b", 2, INF, "info"), R(2, "b", 3, 4), R(3, "a", 5, 6)]
    assert not check_key("k", h).ok


def test_failed_ops_are_ignored():
    h = [W(0, "a", 0, 1), W(1, "b", 2, 3, "fail"), R(2, "a", 4, 5), R(3, "x", 6, 7, "info")]
    assert check_key("k", h).ok


def test_keys_are_checked_independently():
    h = [W(0, "a", 0, 1, key="x"), W(1, "b", 0, 1, key="y"), R(2, "a", 2, 3, key="x"),
         R(3, "a", 2, 3, key="y")]
    res = {r.key: r.ok for r in check_history(h)}
    assert res == {"x": True, "y": False}


def test_random_linearizable_histories_pass():
    # Simulate an atomic register with random overlapping intervals around
    # each operation's linearization point.
    rng = random.Random(3)
    for trial in range(30):
        t, state, ops = 0.0, None, []
        for i in range(60):
            t += rng.uniform(0.1, 1.0)
            call = t - rng.uniform(0, 2.0)
            ret = t + rng.uniform(0, 2.0)
            if rng.random() < 0.5:
                state = f"v{trial}_{i}"
                ops.append(W(i % 5, state, call, ret))
            else:
                ops.append(R(i % 5, state, call, ret))
        assert check_key("k", ops).ok, trial


def test_random_histories_with_injected_stale_read_fail():
    rng = random.Random(5)
    caught = 0
    for trial in range(30):
        t, state, ops, old = 0.0, None, [], []
        for i in range(40):
            t += 1.0
            if rng.random() < 0.6:
                old.append(state)
                state = f"v{trial}_{i}"
                ops.append(W(0, state, t - 0.2, t + 0.2))
            else:
                ops.append(R(1, state, t - 0.2, t + 0.2))
        stale = next((s for s in reversed(old[:-2]) if s is not None), None)
        if stale is None:
            continue
        ops.append(R(2, stale, t + 5, t + 6))   # strictly after everything
        assert not check_key("k", ops).ok
        caught += 1
    assert caught > 10
