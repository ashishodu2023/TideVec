#!/usr/bin/env python3
"""Check that every version string in the repo agrees (and matches a release tag).

    python scripts/check_release_versions.py                 # all files agree
    python scripts/check_release_versions.py --tag v0.3.0    # ... and equal the tag

Exits 1 and lists the offending files on mismatch.
"""

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

TS_PACKAGES = ["tidevec", "ai-sdk", "n8n-nodes-tidevec"]


def json_path(*keys):
    """Extractor for a value inside a JSON file (keys may be dict keys or list indexes)."""
    def get(text):
        node = json.loads(text)
        for k in keys:
            node = node[k]
        return node
    get.label = "/".join(str(k) for k in keys)
    return get


# (file, regex with one group capturing the version or a json_path extractor[, label])
SOURCES = [
    ("include/tidevec/api/rest_server.hpp", r'#define TIDEVEC_SERVER_VERSION "([^"]+)"'),
    ("CMakeLists.txt", r"project\(TideVec VERSION ([0-9.]+)"),
    # Python
    ("sdk/python/pyproject.toml", r'(?m)^version\s*=\s*"([^"]+)"'),
    ("sdk/python/tidevec/__init__.py", r'__version__\s*=\s*"([^"]+)"'),
    ("sdk/python/pyproject.toml", r'"tidevec-embedded>=([^"]+)"', "tidevec-embedded>="),
    ("sdk/python-embedded/pyproject.toml", r'(?m)^version\s*=\s*"([^"]+)"'),
    ("packaging/pypi/tidevec-mcp/pyproject.toml", r'(?m)^version\s*=\s*"([^"]+)"'),
    ("packaging/pypi/tidevec-mcp/pyproject.toml", r'"tidevec\[[^\]]*\]==([^"]+)"', "tidevec[...]=="),
    # Java / C++
    ("sdk/java/pom.xml", r"<artifactId>tidevec-java</artifactId>\s*<version>([^<]+)</version>"),
    ("sdk/cpp/CMakeLists.txt", r"VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)"),
    ("sdk/cpp/conanfile.py", r'version\s*=\s*"([^"]+)"'),
    ("sdk/cpp/vcpkg.json", r'"version"\s*:\s*"([^"]+)"'),
    # TypeScript (package.json, the dependency on the client, the lockfile)
    *[(f"sdk/typescript/packages/{p}/package.json", json_path("version")) for p in TS_PACKAGES],
    *[(f"sdk/typescript/packages/{p}/package.json", r'"tidevec"\s*:\s*"\^?([^"]+)"', "tidevec dependency")
      for p in TS_PACKAGES if p != "tidevec"],
    *[("sdk/typescript/package-lock.json", json_path("packages", f"packages/{p}", "version"))
      for p in TS_PACKAGES],
    # Docker, Helm, MCP registry, Homebrew
    ("docker/cpu/Dockerfile", r"ARG VERSION=(\S+)"),
    ("docker/gpu/Dockerfile", r"ARG VERSION=(\S+)"),
    ("docker/tpu/Dockerfile", r"ARG VERSION=(\S+)"),
    ("deploy/helm/tidevec/Chart.yaml", r'(?m)^version:\s*"?([^"\s]+)"?', "version"),
    ("deploy/helm/tidevec/Chart.yaml", r'(?m)^appVersion:\s*"?([^"\s]+)"?', "appVersion"),
    ("packaging/mcp/server.json", json_path("version")),
    ("packaging/mcp/server.json", json_path("packages", 0, "version")),
    ("packaging/mcp/Dockerfile", r"ARG TIDEVEC_MCP_VERSION=(\S+)"),
    ("packaging/homebrew/tidevec.rb", r'url "[^"]*/v([0-9][^"/]*)\.tar\.gz"'),
]


def extract(text, how):
    if callable(how):
        try:
            return str(how(text))
        except (KeyError, IndexError, TypeError, ValueError):
            return None
    m = re.search(how, text)
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="release tag, e.g. v0.3.0 or sdk/go/v0.3.0")
    args = ap.parse_args()

    found = []
    for rel, how, *note in SOURCES:
        path = ROOT / rel
        note = note[0] if note else getattr(how, "label", "")
        label = f"{rel} [{note}]" if note else rel
        if not path.exists():
            print(f"  skip  {label} (not present)")
            continue
        found.append((label, extract(path.read_text(encoding="utf-8"), how)))

    expected = re.sub(r"^(sdk/go/)?v", "", args.tag) if args.tag else None
    if expected is None:
        versions = [v for _, v in found if v]
        expected = max(set(versions), key=versions.count) if versions else ""

    bad = 0
    for label, v in found:
        ok = v == expected
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {label}: {v or 'version string not found'}")

    if bad:
        print(f"\n{bad} file(s) do not say {expected}")
        return 1
    print(f"\nall version strings say {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
