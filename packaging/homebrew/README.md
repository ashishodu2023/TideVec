# Homebrew

`tidevec.rb` builds `tidevec-server` from the GitHub source tarball with
CMake + Ninja (no network access needed during the build; all C++
dependencies are vendored under `include/`).

## Publishing (release worker)

1. Create the tap repo `ashishodu2023/homebrew-tap` (once).
2. After tagging `vX.Y.Z`:

   ```bash
   packaging/homebrew/update_formula.sh X.Y.Z     # fills in url + sha256
   cp packaging/homebrew/tidevec.rb ../homebrew-tap/Formula/tidevec.rb
   cd ../homebrew-tap && brew audit --strict --online tidevec && brew test-bot --only-formulae tidevec
   ```

3. Users then run:

   ```bash
   brew install ashishodu2023/tap/tidevec
   brew services start tidevec        # 127.0.0.1:6399, auth off, data in $(brew --prefix)/var/tidevec
   ```

## Notes

- On Intel the CMake default `-march=native` is overridden (`-DTIDEVEC_MARCH=`),
  because bottles must run on any supported CPU. Apple Silicon needs no flag.
- TLS is not enabled (`-DTIDEVEC_TLS=ON` would add an `openssl@3` dependency);
  the service binds to loopback only.
- `brew test` starts the server on a free port, checks `/health` reports the
  formula version, and does a create → upsert → search round trip.
