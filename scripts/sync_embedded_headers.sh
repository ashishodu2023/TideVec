#!/usr/bin/env bash
# Vendor the C++ headers into sdk/python-embedded so its sdist is self-contained.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/sdk/python-embedded/include"
rm -rf "$dest"
mkdir -p "$dest"
cp -r "$root/include/tidevec" "$dest/"
echo "Synced headers into $dest"
