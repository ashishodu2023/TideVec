#!/usr/bin/env bash
# Point packaging/homebrew/tidevec.rb at a released tag:
#   packaging/homebrew/update_formula.sh 0.3.0
# Downloads the GitHub source tarball, then rewrites url and sha256 in place.
set -euo pipefail

version="${1:?usage: $0 <version>}"
formula="$(cd "$(dirname "$0")" && pwd)/tidevec.rb"
url="https://github.com/ashishodu2023/TideVec/archive/refs/tags/v${version}.tar.gz"

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT
curl -fsSL "$url" -o "$tmp"
if command -v sha256sum >/dev/null; then
    sha="$(sha256sum "$tmp" | cut -d' ' -f1)"
else
    sha="$(shasum -a 256 "$tmp" | cut -d' ' -f1)"
fi

python3 - "$formula" "$url" "$sha" <<'PY'
import re, sys
path, url, sha = sys.argv[1:]
src = open(path).read()
src = re.sub(r'(?m)^(\s*url ")[^"]+(")', rf'\g<1>{url}\g<2>', src, count=1)
src = re.sub(r'(?m)^(\s*sha256 ")[0-9a-f]{64}(")', rf'\g<1>{sha}\g<2>', src, count=1)
open(path, "w").write(src)
PY
echo "tidevec.rb -> v${version} (${sha})"
