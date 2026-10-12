#!/usr/bin/env python3
"""Generate the REST API reference (docs/api/rest.md) from the server source.

    python scripts/gen_api_docs.py           # write docs/api/rest.md
    python scripts/gen_api_docs.py --check   # exit 1 if the file is out of date

Reads every `svr_.Get/Post/Put/Delete/Patch(...)` registration under include/tidevec/api/
and, for each route, uses:
  * the `// METHOD /path - description` comment block directly above it
    (`Body:` lines become the request example),
  * the handler body: auth, rate limiting, leader-only writes, tenant checks, status codes,
  * JSON keys read by the handler, by the matching `_apply_write` branch and by any
    `*_from_json` helper it calls (request fields with defaults),
  * keys of the `ok({...})` / `return {...}` object it builds (response fields).

Standard library only, so it runs in CI before anything is built.
"""

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "include" / "tidevec" / "api"
OUT = ROOT / "docs" / "api" / "rest.md"

ROUTE_RE = re.compile(
    r'\bsvr_?\.(Get|Post|Put|Delete|Patch)\(\s*(R"\((.*?)\)"|"((?:[^"\\]|\\.)*)")', re.S)
KEY_RE = re.compile(
    r'\b(\w+)\s*(?:\.\s*(at|value|contains)\s*\(\s*"(\w+)"\s*(?:,\s*([^;]*?))?\)|\[\s*"(\w+)"\s*\])')
FROM_JSON_CALL_RE = re.compile(r'\b(\w+_from_json)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)')
ERROR_CODE_RE = re.compile(r'_error\(\s*res\s*,(.*?)(?:,\s*(\d{3}))?\s*\)\s*;', re.S)
JSON_STATUS_RE = re.compile(r'_json_response\(\s*res\s*,.*?,\s*(\d{3})\s*\)\s*;', re.S)
WRITE_RE = re.compile(r'_write\(\s*res\s*,\s*"(\w+)"\s*,[^;]*?(?:,\s*(\d{3}))?\s*\)\s*;', re.S)

STATUS_TEXT = {
    200: "OK", 201: "Created", 204: "No Content", 400: "Bad request", 401: "Missing or bad API key",
    403: "Forbidden (tenant, quota, or URL not allowed)", 404: "Not found", 409: "Conflict",
    421: "Not the Raft leader (cluster mode)", 429: "Rate limited", 500: "Internal error",
    501: "Not supported in this mode", 503: "Unavailable / leadership changed",
}

SECTIONS = [
    ("Server", lambda p: p in ("/health", "/v1/info", "/metrics")),
    ("Cluster", lambda p: p.startswith(("/v1/cluster", "/v1/admin/cluster"))),
    ("Aliases", lambda p: "/aliases" in p),
    ("Model migration (DriftBridge)", lambda p: "/drift" in p),
    ("Search", lambda p: re.search(r"/(search|query|hybrid)", p) is not None),
    ("Vectors", lambda p: re.search(r"/(upsert|delete|vectors|points|get|scroll|ids|count)\b", p)
     is not None),
    ("Edges", lambda p: "/edges" in p),
    ("Collections", lambda p: p.startswith("/v1/collections")),
    ("Testing only", lambda p: p.startswith("/v1/admin/fault")),
    ("Admin and backups", lambda p: p.startswith("/v1/admin")),
    ("Internal (Raft RPC)", lambda p: p.startswith("/raft")),
]

# Fallback one-liners for routes whose comment block has no description.
# Keyed by "METHOD path-shape"; routes not listed here still render from their comments.
SUMMARIES = {
    "GET /health": "Liveness, version and server mode flags. No auth.",
    "GET /v1/info": "Version and feature manifest. No auth.",
    "GET /metrics": "Prometheus metrics (text format). No auth.",
    "GET /v1/cluster/status": "Raft role, term, leader and commit/applied indexes, or "
                              "`{\"mode\":\"standalone\"}`.",
    "GET /v1/collections": "List collections.",
    "GET /v1/collections/*": "Collection stats and supersession settings.",
    "DELETE /v1/collections/*": "Drop a collection and its data.",
    "POST /v1/collections/*/upsert": "Insert or replace vectors (one or a batch).",
    "POST /v1/collections/*/delete": "Delete vectors by ids and/or filter.",
    "POST /v1/collections/*/search": "Vector, keyword or hybrid search with recency and "
                                     "supersession.",
    "POST /v1/collections/*/edges": "Add typed edges (`CAUSES`, `CONTRADICTS`, `UPDATES`, ...).",
    "PUT /v1/collections/*/temporal": "Update the collection's half-life and recency blend.",
    "POST /v1/collections/*/drift/start": "Start a DriftBridge model migration "
                                          "(standalone servers only).",
    "DELETE /v1/aliases/*": "Remove an alias.",
    "PUT /v1/aliases/*": "Create an alias or atomically repoint it to another collection.",
    "POST /v1/collections": "Create a collection (temporal, agent, supersession settings).",
    "POST /v1/collections/*/get": "Fetch vectors by id, in request order; unknown ids are "
                                  "listed in `missing`.",
    "GET /v1/collections/*/ids": "List ids in id order, paged with a cursor.",
    "POST /v1/collections/*/ids": "List ids in id order, paged with a cursor (filter, as_of).",
    "POST /v1/collections/*/count": "Count vectors, optionally filtered and as of a time.",
    "POST /raft/append_entries": "Raft AppendEntries RPC between cluster nodes.",
    "POST /raft/request_vote": "Raft RequestVote RPC between cluster nodes.",
    "POST /raft/install_snapshot": "Raft InstallSnapshot RPC (chunked) to a lagging follower.",
    "POST /v1/admin/cluster/add_node": "Add a node to the Raft cluster (experimental).",
    "POST /v1/admin/cluster/remove_node": "Remove a node from the Raft cluster (experimental).",
    "GET /v1/admin/fault": "Fault-injection state. Only with `--enable-fault-injection` "
                           "(tests).",
    "POST /v1/admin/fault": "Set fault injection. Only with `--enable-fault-injection` "
                            "(tests).",
}

PARAM_NAMES = {"collections": "name", "aliases": "alias", "vectors": "id", "points": "id",
               "get": "id", "trace": "query_id", "backups": "snapshot"}


@dataclass
class Field:
    name: str
    required: bool
    default: Optional[str]
    source: str


@dataclass
class Route:
    method: str
    path: str
    raw_path: str
    file: str
    line: int
    summary: str = ""
    description: List[str] = field(default_factory=list)
    body_example: List[str] = field(default_factory=list)
    auth: str = "none"
    rate_limited: bool = False
    leader_only: bool = False
    tenant_checked: bool = False
    admin_only: bool = False
    statuses: Dict[int, str] = field(default_factory=dict)
    request: List[Field] = field(default_factory=list)
    response: List[str] = field(default_factory=list)
    write_op: Optional[str] = None
    query: List[str] = field(default_factory=list)


# ---------------------------------------------------------------- helpers

def match_bracket(text: str, start: int, open_ch: str = "{", close_ch: str = "}") -> int:
    """Index just past the bracket matching text[start] (skips string literals)."""
    depth, i, n = 0, start, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            nl = text.find("\n", i)
            i = nl if nl >= 0 else n
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = end + 2 if end >= 0 else n
            continue
        if c == '"':
            if text.startswith('R"(', i - 1):
                end = text.find(')"', i)
                i = end + 2 if end >= 0 else n
                continue
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
        elif c == "'":
            i += 1
            while i < n and text[i] != "'":
                i += 2 if text[i] == "\\" else 1
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def strip_line_comments(code: str) -> str:
    return re.sub(r"//[^\n]*", "", code)


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


SENTINELS = {("temporal_blend", "-1.0"): "collection setting"}


def clean_default(raw: Optional[str], key: str = "") -> Optional[str]:
    if raw is None:
        return None
    d = raw.strip()
    d += ")" * (d.count("(") - d.count(")"))
    d = re.sub(r"^std::string\((.*)\)$", r"\1", d)
    d = re.sub(r"^\w+(?:_t)?\{(.*)\}$", r"\1", d)
    d = re.sub(r"^(-?\d+(?:\.\d+)?)(?:f|UL|L|LL|ULL|u)$", r"\1", d)
    if (key, d) in SENTINELS:
        return SENTINELS[(key, d)]
    if not re.fullmatch(r'-?\d+(?:\.\d+)?|"[^"]*"|true|false|null|nullptr', d):
        return "server default"
    return d


def human_path(raw: str, comment_path: Optional[str]) -> str:
    p = raw.lstrip("^").rstrip("$")
    groups = re.findall(r"\((?!\?)[^)]*\)", p)
    if comment_path:
        cp_params = re.findall(r"\{(\w+)\}", comment_path)
        if len(cp_params) == len(groups):
            for g, name in zip(groups, cp_params):
                p = p.replace(g, "{" + name + "}", 1)
            return p
    out, n = [], 0
    for seg in p.split("/"):
        if re.fullmatch(r"\((?!\?)[^)]*\)", seg):
            prev = out[-1] if out else ""
            n += 1
            out.append("{" + PARAM_NAMES.get(prev, f"param{n}") + "}")
        else:
            out.append(seg)
    return "/".join(out)


# ---------------------------------------------------------------- parsing

FUNC_DEF_RE = re.compile(r"\b(\w+)\s*\(([^(){};]*)\)\s*(?:const\s*)?\{")


def split_args(s: str) -> List[str]:
    args, depth, cur = [], 0, ""
    for c in s:
        if c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth -= 1
        if c == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += c
    if cur.strip():
        args.append(cur.strip())
    return args


def json_helpers(sources: Dict[str, str]) -> Dict[str, Tuple[int, str, str]]:
    """name -> (index of the json param, its name, body) for functions taking `const json&`."""
    helpers = {}
    for text in sources.values():
        for m in FUNC_DEF_RE.finditer(text):
            name = m.group(1)
            if name in ("if", "for", "while", "switch", "catch", "_json_response", "ok", "err"):
                continue
            params = split_args(m.group(2))
            for idx, p in enumerate(params):
                pm = re.fullmatch(r"const\s+json\s*&\s*(\w+)(?:\s*=.*)?", p)
                if pm:
                    body_start = m.end() - 1
                    helpers[name] = (idx, pm.group(1),
                                     text[body_start:match_bracket(text, body_start)])
                    break
    return helpers


def member_helpers(sources: Dict[str, str]) -> Dict[str, str]:
    """name -> body of member functions taking (const httplib::Request& req, Response& res...)."""
    out = {}
    for text in sources.values():
        for m in re.finditer(r"\b(_\w+)\s*\(\s*const\s+httplib::Request\s*&\s*\w+\s*,"
                             r"\s*httplib::Response\s*&\s*\w+[^)]*\)\s*\{", text):
            start = m.end() - 1
            out[m.group(1)] = text[start:match_bracket(text, start)]
    return out


def find_calls(code: str, name: str) -> List[List[str]]:
    calls = []
    for m in re.finditer(r"\b" + re.escape(name) + r"\s*\(", code):
        start = m.end() - 1
        end = match_bracket(code, start, "(", ")")
        calls.append(split_args(code[start + 1:end - 1]))
    return calls


def apply_write_branches(sources: Dict[str, str]) -> Dict[str, str]:
    """op -> code of the `if (op == "op") { ... }` branch in _apply_write."""
    branches = {}
    for text in sources.values():
        m = re.search(r"json\s+_apply_write\s*\([^)]*\)\s*\{", text)
        if not m:
            continue
        start = m.end() - 1
        body = text[start:match_bracket(text, start)]
        ops = list(re.finditer(r'if\s*\(\s*op\s*==\s*"(\w+)"\s*\)\s*\{', body))
        shared_prefix_end = ops[0].start() if ops else 0
        for om in ops:
            b0 = om.end() - 1
            branches[om.group(1)] = body[b0:match_bracket(body, b0)]
        # Code between the first op-branch and the next one (e.g. `auto& col = registry_.get`)
        branches["__prefix__"] = body[:shared_prefix_end]
    return branches


def extract_fields(code: str, var: str, source: str, helpers, prefix: str = "",
                   depth: int = 0) -> List[Field]:
    code = strip_line_comments(code)
    fields: List[Field] = []
    nested_vars: Dict[str, str] = {}
    # loop variables over arrays: for (const auto& item : j["vectors"])
    for lm in re.finditer(r'for\s*\(\s*(?:const\s+)?auto\s*&?\s*(\w+)\s*:\s*' + re.escape(var)
                          + r'(?:\s*\.\s*at\s*\(\s*"(\w+)"\s*\)|\s*\[\s*"(\w+)"\s*\])', code):
        nested_vars[lm.group(1)] = (lm.group(2) or lm.group(3)) + "[]."
    for km in KEY_RE.finditer(code):
        obj, fn, key, default, idx_key = km.groups()
        key = key or idx_key
        if obj == var:
            name = prefix + key
        elif obj in nested_vars:
            name = prefix + nested_vars[obj] + key
        else:
            continue
        required = fn == "at"
        fields.append(Field(name, required,
                            clean_default(default, key) if fn == "value" else None, source))
    if depth < 4:
        for fn_name, (pidx, h_var, h_body) in helpers.items():
            for args in find_calls(code, fn_name):
                if pidx >= len(args):
                    continue
                arg = args[pidx]
                am = re.fullmatch(re.escape(var) + r'\s*(?:\[\s*"(\w+)"\s*\]|\.at\(\s*"(\w+)"\s*\))',
                                  arg)
                if am:
                    sub_prefix = prefix + (am.group(1) or am.group(2)) + "."
                elif arg == var:
                    sub_prefix = prefix
                elif arg in nested_vars:
                    sub_prefix = prefix + nested_vars[arg]
                else:
                    lam = re.search(r'auto\s+\w+\s*=\s*\[[^\]]*\]\s*\(\s*const\s+json\s*&\s*'
                                    + re.escape(arg) + r'\s*\)', code)
                    arr = re.search(r'for\s*\([^)]*:\s*' + re.escape(var)
                                    + r'\s*\[\s*"(\w+)"\s*\]', code)
                    if not (lam and arr):
                        continue
                    sub_prefix = prefix + arr.group(1) + "[]."
                src = source if fn_name.startswith("_") else f"`{fn_name}`"
                fields += extract_fields(h_body, h_var, src, helpers, sub_prefix, depth + 1)
    return fields


def response_keys(code: str) -> List[str]:
    keys: List[str] = []
    code = strip_line_comments(code)
    for m in re.finditer(r"\b(?:ok|_json_response)\s*\(", code):
        start = m.end() - 1
        chunk = code[start:match_bracket(code, start, "(", ")")]
        keys += top_level_keys(chunk)
    for m in re.finditer(r"\breturn\s*(?:json\s*)?\{", code):
        start = m.end() - 1
        keys += top_level_keys(code[start:match_bracket(code, start)])
    seen, out = set(), []
    for k in keys:
        if k not in seen and k not in ("status", "data"):
            seen.add(k)
            out.append(k)
    return out


def top_level_keys(chunk: str) -> List[str]:
    """Keys of `{"k", v}` pairs at the outermost initializer-list level."""
    keys, depth, i = [], 0, 0
    first_brace = chunk.find("{")
    if first_brace < 0:
        return keys
    base = None
    while i < len(chunk):
        c = chunk[i]
        if c == '"':
            j = i + 1
            while j < len(chunk) and chunk[j] != '"':
                j += 2 if chunk[j] == "\\" else 1
            if base is not None and depth == base + 1 and chunk[i - 1] == "{":
                keys.append(chunk[i + 1:j])
            i = j + 1
            continue
        if c == "{":
            depth += 1
            if base is None:
                base = depth if chunk[i + 1:].lstrip().startswith("{") else depth - 1
        elif c == "}":
            depth -= 1
        i += 1
    return [k for k in keys if re.fullmatch(r"\w+", k)]


HEADER_RE = re.compile(r"^\s*(GET|POST|PUT|DELETE|PATCH)\s+(/\S*)\s*(.*)$")


def clean_summary(rest: str) -> str:
    rest = rest.strip()
    rest = re.sub(r"^[-—–:]+\s*", "", rest)
    rest = re.sub(r"^\((.*)\)$", r"\1", rest)
    if rest.startswith("→"):
        rest = "Returns `" + rest[1:].strip() + "`"
    elif rest.startswith(("{", "[")):
        rest = "Body: `" + rest + "`"
    return rest


def path_shape(path: str) -> str:
    """'/v1/collections/{name}/ids?limit=1' and '/v1/collections/([^/]+)/ids' -> same shape."""
    p = path.split("?")[0].lstrip("^").rstrip("$")
    p = re.sub(r"\{\w+\}", "*", p)
    p = re.sub(r"\((?!\?)[^)]*\)", "*", p)
    return p


def comment_headers(lines: List[str]) -> List[Tuple[str, str, str]]:
    """Every `// METHOD /path rest` line in a file."""
    out = []
    for line in lines:
        s = line.strip()
        if s.startswith("//"):
            hm = HEADER_RE.match(s[2:])
            if hm:
                out.append((hm.group(1), hm.group(2), hm.group(3)))
    return out


def parse_comment_block(lines: List[str], idx: int, method: str
                        ) -> Tuple[Optional[str], str, List[str], List[str]]:
    """Comment lines directly above line index idx (0-based)."""
    block: List[str] = []
    i = idx - 1
    while i >= 0 and lines[i].strip().startswith("//"):
        block.insert(0, lines[i].strip()[2:].rstrip())
        i -= 1
    block = [b for b in block if not re.fullmatch(r"\s*[-=]{4,}\s*", b)]
    comment_path, summary, desc, body = None, "", [], []
    headers = [HEADER_RE.match(b) for b in block]
    if not any(h and h.group(1) == method for h in headers):
        headers = [None] * len(block)   # block documents other routes only
    in_body = False
    for raw, hm in zip(block, headers):
        line = raw[1:] if raw.startswith(" ") else raw
        if hm:
            if hm.group(1) == method and comment_path is None:
                comment_path = hm.group(2)
                summary = clean_summary(hm.group(3))
            continue
        bm = re.match(r"\s*Body:\s*(.*)$", line)
        if bm:
            in_body = True
            if bm.group(1):
                body.append(bm.group(1))
            continue
        if in_body and (line.startswith("  ") or line.strip().startswith(("or ", "}", "\"")))\
                and line.strip():
            body.append(line.rstrip())
            continue
        in_body = False
        if line.strip():
            desc.append(line.strip())
        elif desc and desc[-1] != "":
            desc.append("")
    while desc and desc[-1] == "":
        desc.pop()
    return comment_path, summary, desc, body


def parse_routes() -> List[Route]:
    sources = {p.name: p.read_text(encoding="utf-8")
               for p in sorted(SRC_DIR.glob("*.hpp")) if p.name not in ("httplib.h", "json.hpp")}
    helpers = json_helpers(sources)
    members = member_helpers(sources)
    branches = apply_write_branches(sources)
    routes: List[Route] = []

    for fname, text in sources.items():
        lines = text.split("\n")
        headers = comment_headers(lines)
        for m in ROUTE_RE.finditer(text):
            method = m.group(1).upper()
            raw = m.group(3) if m.group(3) is not None else m.group(4)
            ln = line_of(text, m.start())
            comment_line = ln - 1
            vm = re.match(r"\s*,\s*(\w+)\s*\)", text[m.end():])
            if vm:   # handler stored in a variable: `auto h = [this](...) {...}; svr_.Get(p, h);`
                dm = None
                for dm in re.finditer(r"\bauto\s+" + re.escape(vm.group(1)) + r"\s*=\s*\[", text[:m.start()]):
                    pass
                if not dm:
                    continue
                lam = text.find("{", dm.end())
                handler = text[dm.start():match_bracket(text, lam)]
                comment_line = line_of(text, dm.start()) - 1
            else:
                lam = text.find("{", m.end())
                handler = text[m.end():match_bracket(text, lam) if lam >= 0 else m.end()]
            comment_path, summary, desc, body = parse_comment_block(lines, comment_line, method)
            if comment_path is None:   # a shared comment block elsewhere may list this route
                for hmethod, hpath, rest in headers:
                    if hmethod == method and path_shape(hpath) == path_shape(raw):
                        comment_path, summary = hpath, clean_summary(rest)
                        break
            if comment_path:
                comment_path = comment_path.split("?")[0]
            r = Route(method, human_path(raw, comment_path), raw, fname, ln,
                      summary, desc, body)
            code = strip_line_comments(handler)
            r.query = sorted(set(re.findall(r'req\.(?:get_param_value|has_param)\(\s*"(\w+)"', code)))
            for name, mbody in members.items():   # shared helpers such as _read(req, res, fn)
                if re.search(r"\b" + re.escape(name) + r"\s*\(\s*req\s*,\s*res", code):
                    code += "\n" + strip_line_comments(mbody)

            if "_cluster_auth(" in code:
                r.auth = "cluster secret (`X-TideVec-Cluster-Secret`)"
            elif "_auth(" in code:
                r.auth = "API key"
            r.admin_only = r.auth == "API key" and "_require_admin(" in code
            r.rate_limited = "_rate_limit(" in code
            r.tenant_checked = "_tenant_" in code
            wm = WRITE_RE.search(code)
            if wm:
                r.write_op = wm.group(1)
            r.leader_only = "_follower_redirect(" in code or r.write_op is not None

            ok_code = int(wm.group(2)) if wm and wm.group(2) else 200
            r.statuses[ok_code] = STATUS_TEXT.get(ok_code, "")
            for jm in JSON_STATUS_RE.finditer(code):
                c = int(jm.group(1))
                if c < 300:
                    r.statuses[c] = STATUS_TEXT.get(c, "")
            for em in ERROR_CODE_RE.finditer(code):
                c = int(em.group(2)) if em.group(2) else 400
                r.statuses[c] = STATUS_TEXT.get(c, "")
            if re.search(r"registry_\.(get|acquire)\(", code) or (r.write_op and r.write_op not in ("create",)):
                r.statuses[404] = "Collection not found"
            if "json::parse(" in code:
                r.statuses.setdefault(400, STATUS_TEXT[400])
            if r.auth == "API key":
                r.statuses[401] = STATUS_TEXT[401]
            if r.tenant_checked:
                r.statuses.setdefault(403, STATUS_TEXT[403])
            if r.admin_only:
                r.statuses[403] = ("Not the admin key, or " + STATUS_TEXT[403][0].lower()
                                   + STATUS_TEXT[403][1:]) if r.tenant_checked else \
                                  "Not the admin API key (tenant keys)"
            if r.rate_limited:
                r.statuses[429] = STATUS_TEXT[429]
            if r.leader_only:
                r.statuses[421] = STATUS_TEXT[421]
                if r.write_op:
                    r.statuses[503] = "Write not confirmed (cluster mode)"

            body_var = None
            pm = re.search(r"\b(\w+)\s*=[^;=]*?json::parse\(\s*req\.body\s*\)", code)
            if pm:
                body_var = pm.group(1)
            fields: List[Field] = []
            if body_var:
                fields += extract_fields(code, body_var, "handler", helpers)
            if r.write_op and r.write_op in branches:
                fields += extract_fields(branches[r.write_op], "j", "handler", helpers)
            seen = {}
            for f in fields:
                if f.name not in seen:
                    seen[f.name] = f
                else:
                    s = seen[f.name]
                    s.required = s.required or f.required
                    s.default = s.default or f.default
            r.request = list(seen.values())

            resp_code = code + (branches.get(r.write_op, "") if r.write_op else "")
            r.response = response_keys(resp_code)
            routes.append(r)
    return routes


# ---------------------------------------------------------------- rendering

def anchor(r: Route) -> str:
    return re.sub(r"[^a-z0-9]+", "-", f"{r.method} {r.path}".lower()).strip("-")


def display_summary(r: Route) -> str:
    override = SUMMARIES.get(f"{r.method} {path_shape(r.raw_path)}")
    if override:
        return override
    if r.summary and not r.summary.startswith(("Body:", "Returns")):
        return r.summary[0].upper() + r.summary[1:]
    if r.summary:
        return r.summary
    if r.description and not r.description[0].startswith(("→", "{")):
        text = " ".join(r.description).split("  ")[0]
        return re.split(r"(?<=\.)\s", text)[0]
    return ""


def section_of(path: str) -> str:
    for name, pred in SECTIONS:
        if pred(path):
            return name
    return "Other"


def render(routes: List[Route]) -> str:
    out: List[str] = []
    w = out.append
    w("<!-- Generated by scripts/gen_api_docs.py from include/tidevec/api/*.hpp. Do not edit;")
    w("     rerun `python scripts/gen_api_docs.py` after changing a route. -->")
    w("")
    w("# REST API reference")
    w("")
    w("The server listens on port **6399** (HTTP/1.1, JSON). Routes under `/v1/` need an "
      "`X-Api-Key` header when auth is enabled (the default in the Docker image). Every JSON "
      "response has the shape `{\"status\": \"ok\", \"data\": ...}` or "
      "`{\"status\": \"error\", \"error\": \"...\", \"code\": N}`.")
    w("")
    w("This page is generated from the route registrations in "
      "[`include/tidevec/api/rest_server.hpp`]"
      "(https://github.com/ashishodu2023/TideVec/blob/main/include/tidevec/api/rest_server.hpp). "
      "Request fields are the JSON keys the handler reads (with the default it applies); "
      "response fields are the keys of the `data` object it returns.")
    w("")
    w("In [cluster mode](../operations/cluster.md) writes must go to the leader: followers answer "
      "`421` with the leader URL in the `X-TideVec-Leader` header and the `leader` field.")
    w("")

    by_section: Dict[str, List[Route]] = {}
    for r in routes:
        by_section.setdefault(section_of(r.path), []).append(r)
    order = [s for s, _ in SECTIONS] + ["Other"]

    w("## Endpoints")
    w("")
    w("| Method | Path | Auth | Description |")
    w("|---|---|---|---|")
    for s in order:
        for r in by_section.get(s, []):
            flags = []
            if r.rate_limited:
                flags.append("rate-limited")
            if r.leader_only:
                flags.append("leader")
            auth = {"API key": "key", "none": "—"}.get(r.auth, "cluster")
            if r.admin_only:
                auth = "admin key"
            desc = display_summary(r)
            extra = f" ({', '.join(flags)})" if flags else ""
            w(f"| `{r.method}` | [`{r.path}`](#{anchor(r)}) | {auth} | {desc}{extra} |")
    w("")

    for s in order:
        rs = by_section.get(s, [])
        if not rs:
            continue
        w(f"## {s}")
        w("")
        for r in rs:
            w(f"### `{r.method} {r.path}` {{#{anchor(r)}}}")
            w("")
            summary = display_summary(r)
            if summary and not (r.description and summary == re.split(r"(?<=\.)\s", " ".join(r.description))[0]):
                w(summary[0].upper() + summary[1:] + ("" if summary.endswith((".", "`")) else "."))
                w("")
            if r.description:
                w("\n".join(r.description))
                w("")
            facts = [f"**Auth:** {'admin API key' if r.admin_only else r.auth}"]
            if r.rate_limited:
                facts.append("**Rate-limited**")
            if r.leader_only:
                facts.append("**Cluster mode:** leader only")
            if r.write_op:
                facts.append(f"**Replicated write** (`{r.write_op}`)")
            w(" · ".join(facts))
            w("")
            if r.body_example:
                w("Example body:")
                w("")
                w("```json")
                w("\n".join(b.strip() if i == 0 else b for i, b in enumerate(r.body_example)))
                w("```")
                w("")
            if r.request:
                w("| Request field | Required | Default | Read by |")
                w("|---|---|---|---|")
                for f in r.request:
                    if f.default is None:
                        d = ""
                    elif re.fullmatch(r'-?\d+(?:\.\d+)?|"[^"]*"|true|false|null|nullptr',
                                      f.default):
                        d = f"`{f.default}`"
                    else:
                        d = f.default
                    w(f"| `{f.name}` | {'yes' if f.required else ''} | {d} | {f.source} |")
                w("")
            if r.query:
                w("Query parameters: " + ", ".join(f"`{q}`" for q in r.query))
                w("")
            if r.response:
                w("Response `data` fields: " + ", ".join(f"`{k}`" for k in r.response))
                w("")
            codes = ", ".join(f"`{c}` {t}".rstrip() for c, t in sorted(r.statuses.items()))
            w(f"Status codes: {codes}")
            w("")
            w(f"<small>Source: `{r.file}` line {r.line}</small>")
            w("")
    return "\n".join(out).rstrip() + "\n"


def strip_lines(md: str) -> str:
    """Line numbers change with unrelated edits; ignore them in --check."""
    return re.sub(r"line \d+", "line N", md)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail if docs/api/rest.md is stale")
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    args = ap.parse_args()

    routes = parse_routes()
    md = render(routes)
    if args.check:
        current = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if strip_lines(current) != strip_lines(md):
            print(f"{args.out} is out of date; run: python scripts/gen_api_docs.py",
                  file=sys.stderr)
            return 1
        print(f"{args.out} is up to date ({len(routes)} routes)")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(f"wrote {args.out.relative_to(ROOT)} ({len(routes)} routes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
