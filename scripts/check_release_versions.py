#!/usr/bin/env python3
"""Check that every version string in the repo agrees (and matches a release tag).

    python scripts/check_release_versions.py                 # all files agree
    python scripts/check_release_versions.py --tag v0.3.0    # ... and equal the tag

Exits 1 and lists the offending files on mismatch.
"""

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (file, regex with one group capturing the version)
SOURCES = [
    ("include/tidevec/api/rest_server.hpp", r'#define TIDEVEC_SERVER_VERSION "([^"]+)"'),
    ("CMakeLists.txt", r"project\(TideVec VERSION ([0-9.]+)"),
    ("sdk/python/pyproject.toml", r'(?m)^version\s*=\s*"([^"]+)"'),
    ("sdk/python/tidevec/__init__.py", r'__version__\s*=\s*"([^"]+)"'),
    ("sdk/python/pyproject.toml", r'"tidevec-embedded>=([^"]+)"'),
    ("sdk/python-embedded/pyproject.toml", r'(?m)^version\s*=\s*"([^"]+)"'),
    ("sdk/java/pom.xml", r"<artifactId>tidevec-java</artifactId>\s*<version>([^<]+)</version>"),
    ("sdk/cpp/CMakeLists.txt", r"VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)"),
    ("sdk/cpp/conanfile.py", r'version\s*=\s*"([^"]+)"'),
    ("sdk/cpp/vcpkg.json", r'"version"\s*:\s*"([^"]+)"'),
    ("docker/cpu/Dockerfile", r"ARG VERSION=(\S+)"),
    ("docker/gpu/Dockerfile", r"ARG VERSION=(\S+)"),
    ("docker/tpu/Dockerfile", r"ARG VERSION=(\S+)"),
    ("deploy/helm/tidevec/Chart.yaml", r'(?m)^appVersion:\s*"?([^"\s]+)"?'),
    ("sdk/typescript/packages/tidevec/package.json", r'"version"\s*:\s*"([^"]+)"'),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="release tag, e.g. v0.3.0 or sdk/go/v0.3.0")
    args = ap.parse_args()

    found = []
    for rel, pattern in SOURCES:
        path = ROOT / rel
        if not path.exists():
            print(f"  skip  {rel} (not present)")
            continue
        m = re.search(pattern, path.read_text(encoding="utf-8"))
        if not m:
            found.append((rel, None))
            continue
        found.append((rel, m.group(1)))

    expected = re.sub(r"^(sdk/go/)?v", "", args.tag) if args.tag else None
    if expected is None:
        versions = [v for _, v in found if v]
        expected = max(set(versions), key=versions.count) if versions else ""

    bad = 0
    for rel, v in found:
        ok = v == expected
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {rel}: {v or 'version string not found'}")

    if bad:
        print(f"\n{bad} file(s) do not say {expected}")
        return 1
    print(f"\nall version strings say {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
