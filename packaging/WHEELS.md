# tidevec-embedded wheels: musllinux and macOS x86_64

Today `ci.yml` and `release.yml` build `tidevec-embedded` for manylinux
x86_64/aarch64 and macOS arm64 only:

```yaml
CIBW_SKIP: "*-musllinux_* *-manylinux_i686"
CIBW_ARCHS_MACOS: arm64
```

So Alpine users (musl) and Intel Macs fall back to building the sdist,
which needs a C++20 compiler, CMake ≥ 3.20 and pybind11.

## Findings

- **musl**: the headers use only POSIX + C++20 standard library APIs (no
  `execinfo`, `malloc_trim`, `sync_file_range`, `O_DIRECT`, `fallocate`
  or `__GLIBC__` checks), so a musllinux build should need no code changes.
  The risk is runtime behaviour: musl's default thread stack is 128 KiB
  (glibc: 8 MiB). Search recursion, `std::async` workers and the WAL
  thread run on default-sized stacks, so the wheel test must exercise
  search, upsert and recovery. If it crashes, link with
  `-Wl,-z,stack-size=1048576` for musl only.
- **macOS x86_64**: `sdk/python-embedded/CMakeLists.txt` never passes
  `-march`, so wheels are portable as-is. Build on an Intel runner
  (`macos-13`) or cross-compile from `macos-14` with
  `CIBW_ARCHS_MACOS: "x86_64 arm64"`. Cross-compiled x86_64 wheels can
  only be tested under Rosetta; cibuildwheel skips the test with a
  warning unless `CIBW_TEST_SKIP` is configured otherwise.
- **SIMD**: because no `-march` is set, x86_64 wheels compile the
  `__AVX2__` distance kernels out and use scalar/SSE2 code. That is
  correct but slower than the Docker image (x86-64-v3). A follow-up
  could add `__attribute__((target("avx2,fma")))` clones with runtime
  dispatch; that's a code change, not a packaging change.

## Changes for the release worker

Apply to **both** `.github/workflows/ci.yml` (job `embedded-wheels`) and
`.github/workflows/release.yml` (job `wheels-embedded`; also add
`macos-13` to the `install-check` job's `os:` matrix so the Intel wheel
gets a clean-install check, plus an Alpine container job if musl
installs should be gated too):

```diff
     strategy:
       matrix:
-        os: [ubuntu-22.04, ubuntu-24.04-arm, macos-14]
+        os: [ubuntu-22.04, ubuntu-24.04-arm, macos-14, macos-13]
 ...
         env:
           CIBW_BUILD: "cp39-* cp310-* cp311-* cp312-* cp313-*"
-          CIBW_SKIP: "*-musllinux_* *-manylinux_i686"
-          CIBW_ARCHS_MACOS: arm64
+          CIBW_SKIP: "*-manylinux_i686 *-musllinux_i686"
+          CIBW_ARCHS_MACOS: native          # arm64 on macos-14, x86_64 on macos-13
           CIBW_MANYLINUX_X86_64_IMAGE: manylinux_2_28
           CIBW_MANYLINUX_AARCH64_IMAGE: manylinux_2_28
+          CIBW_MUSLLINUX_X86_64_IMAGE: musllinux_1_2
+          CIBW_MUSLLINUX_AARCH64_IMAGE: musllinux_1_2
           CIBW_ENVIRONMENT_MACOS: MACOSX_DEPLOYMENT_TARGET=13.3
```

Notes:

- `MACOSX_DEPLOYMENT_TARGET=13.3` stays: it's what libc++ needs for the
  C++20 features used, on both architectures.
- `macos-13` is GitHub's last Intel image and is being retired. When it
  goes, switch to cross-compiling on `macos-14`:
  `CIBW_ARCHS_MACOS: "x86_64 arm64"` and
  `CIBW_TEST_SKIP: "*-macosx_x86_64"` (or keep the test under Rosetta
  with `arch -x86_64`).
- musllinux roughly doubles Linux wheel build time (two more images × five
  Pythons). If CI time matters, build musllinux only in `release.yml`.
- The existing `CIBW_TEST_COMMAND` (`release_smoke_test.py --embedded`)
  covers create, upsert, supersession-aware search and dimension checks.
  It does not reopen a database, so for musl also run
  `pytest sdk/python/tests -k local` (embedded-mode tests, which include
  persistence across `Local` instances) via `CIBW_TEST_COMMAND_LINUX`.
- Expected new artifacts per release: `*-musllinux_1_2_x86_64.whl`,
  `*-musllinux_1_2_aarch64.whl`, `*-macosx_13_0_x86_64.whl` (cibuildwheel
  rounds 13.3 to `13_0` in the tag).

## Not changed here

`ci.yml`/`release.yml` belong to the release worker, so these changes are
documented rather than applied. Nothing has been built or published.
