"""
Linearizability checker for a set of independent registers (stdlib only).

The search is the Wing & Gong algorithm with Lowe's memoisation, as used
by Porcupine (Athalye) and Knossos (Kingsbury): call/return events sit in
a doubly linked list ordered by time; the checker repeatedly tries to
linearize a pending call whose return has not yet been passed, caching
(set of linearized ops, model state) pairs, and backtracks when it
reaches a return event. Registers are checked independently
(P-compositionality: a history is linearizable iff each key's
sub-history is).

History conventions (see Op):
  · status "ok":   the operation took effect (writes) / returned (reads)
  · status "fail": the operation definitely did not take effect -> dropped
  · status "info": the outcome is unknown (timeout, crash, 503 after the
    entry was appended). Info writes may take effect at any time after
    their call, so their return time is +inf; info reads are dropped.

Two sound reductions keep info writes from blowing up the search
(values written are unique per history):
  · an info write whose value no read observed can always be linearized
    after every other operation, so it is removed;
  · an info write whose value was observed must linearize before the
    earliest return among the reads that observed it, so that becomes
    its return time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

INF = float("inf")


@dataclass
class Op:
    client: int
    kind: str              # "write" | "read"
    key: str
    value: object          # value written / value read (None = absent)
    call: float
    ret: float             # INF if never returned
    status: str = "ok"     # ok | fail | info
    meta: dict = field(default_factory=dict)


@dataclass
class KeyResult:
    key: str
    ok: bool
    n_ops: int
    reason: str = ""
    context: list = field(default_factory=list)
    unknown: bool = False  # search budget exhausted


def prepare(ops: list[Op]) -> list[Op]:
    """Applies the history conventions and reductions above."""
    kept = [o for o in ops if o.status != "fail" and not (o.kind == "read" and o.status != "ok")]
    first_seen: dict[object, float] = {}
    for o in kept:
        if o.kind == "read" and o.value is not None:
            first_seen[o.value] = min(first_seen.get(o.value, INF), o.ret)
    out = []
    for o in kept:
        if o.kind == "write" and o.status == "info":
            if o.value not in first_seen:
                continue
            o = Op(o.client, o.kind, o.key, o.value, o.call, max(o.call, first_seen[o.value]),
                   "info", o.meta)
        out.append(o)
    return out


class _Entry:
    __slots__ = ("is_call", "op", "id", "time", "match", "prev", "next")

    def __init__(self, is_call, op, oid, time):
        self.is_call = is_call
        self.op = op
        self.id = oid
        self.time = time
        self.match = None
        self.prev = None
        self.next = None


def _step(state, op: Op):
    if op.kind == "write":
        return True, op.value
    return state == op.value, state


def check_key(key: str, ops: list[Op], initial=None, budget: int = 5_000_000) -> KeyResult:
    ops = prepare(ops)
    n = len(ops)
    if n == 0:
        return KeyResult(key, True, 0)
    written = {o.value for o in ops if o.kind == "write"}
    for o in ops:
        if o.kind == "read" and o.value is not None and o.value not in written:
            return KeyResult(key, False, n, f"read returned a value that was never written: {o.value!r}",
                             [_fmt(o)])

    events = []
    for i, o in enumerate(ops):
        c = _Entry(True, o, i, o.call)
        r = _Entry(False, o, i, o.ret)
        c.match = r
        events.append(c)
        events.append(r)
    # Calls before returns at equal times: such ops count as concurrent.
    events.sort(key=lambda e: (e.time, 0 if e.is_call else 1))
    head = _Entry(False, None, -1, -INF)
    prev = head
    for e in events:
        prev.next = e
        e.prev = prev
        prev = e

    def lift(e):
        e.prev.next = e.next
        if e.next:
            e.next.prev = e.prev
        m = e.match
        m.prev.next = m.next
        if m.next:
            m.next.prev = m.prev

    def unlift(e):
        m = e.match
        m.prev.next = m
        if m.next:
            m.next.prev = m
        e.prev.next = e
        if e.next:
            e.next.prev = e

    state = initial
    linearized = 0
    cache: set = set()
    calls: list = []
    entry = head.next
    steps = 0
    deepest = 0
    deepest_entry = None
    while head.next is not None:
        steps += 1
        if steps > budget:
            return KeyResult(key, True, n, "search budget exhausted", unknown=True)
        if entry.is_call:
            ok, new_state = _step(state, entry.op)
            if ok:
                new_lin = linearized | (1 << entry.id)
                k = (new_lin, new_state)
                if k not in cache:
                    cache.add(k)
                    calls.append((entry, state))
                    state = new_state
                    linearized = new_lin
                    lift(entry)
                    if len(calls) > deepest:
                        deepest = len(calls)
                        deepest_entry = head.next
                    entry = head.next
                    continue
            entry = entry.next
        else:
            if not calls:
                return KeyResult(key, False, n, _explain(deepest, deepest_entry), _context(deepest_entry))
            entry, state = calls.pop()
            linearized &= ~(1 << entry.id)
            unlift(entry)
            entry = entry.next
    return KeyResult(key, True, n)


def _fmt(o: Op) -> str:
    ret = "inf" if o.ret == INF else f"{o.ret:.6f}"
    return (f"client={o.client} {o.kind}({o.value!r}) [{o.call:.6f}, {ret}] status={o.status}"
            + (f" {o.meta}" if o.meta else ""))


def _explain(deepest, e) -> str:
    return (f"not linearizable: at most {deepest} operations could be linearized; "
            f"no order explains the operations after that point")


def _context(e, limit: int = 12) -> list:
    out = []
    while e is not None and len(out) < limit:
        if e.is_call:
            out.append(_fmt(e.op))
        e = e.next
    return out


def check_history(ops: list[Op], budget_per_key: int = 5_000_000) -> list[KeyResult]:
    by_key: dict[str, list[Op]] = {}
    for o in ops:
        by_key.setdefault(o.key, []).append(o)
    return [check_key(k, v, budget=budget_per_key) for k, v in sorted(by_key.items())]
