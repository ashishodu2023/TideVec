# Releasing

Maintainer notes for cutting a TideVec release. Everything is driven by
[`.github/workflows/release.yml`](https://github.com/ashishodu2023/TideVec/blob/main/.github/workflows/release.yml),
which runs on tags matching `v*` and publishes only for tags.

## Checklist

1. Bump every version string, then check them:

    ```bash
    python scripts/check_release_versions.py --tag v0.3.0
    ```

    It compares the server version (`TIDEVEC_SERVER_VERSION` in `rest_server.hpp`), the CMake project
    version, both Python packages (and the `tidevec-embedded>=` pin), the Java POM, the C++ SDK
    (CMake, Conan, vcpkg), the Docker `VERSION` args, the Helm chart `appVersion` and the TypeScript
    package. CI runs the same check on every push.

2. Move the `[Unreleased]` notes in `CHANGELOG.md` under `## [X.Y.Z] - YYYY-MM-DD`. The release job
   fails if the section is missing, and uses it as the GitHub release notes.
3. Regenerate the API reference if routes changed: `python scripts/gen_api_docs.py`
   (CI fails on drift via `--check`).
4. Tag and push: `git tag v0.3.0 && git push origin v0.3.0`. A tag with a suffix (`v0.4.0-rc1`) is
   published as a pre-release and does not move `latest`, `cpu` or the `X.Y` Docker tags.

## What the workflow does

| Job | Output |
|---|---|
| `versions` | version check and CHANGELOG section check |
| `ci` | the full CI workflow (C++ tests, SDK tests, docs) |
| `dist-python` | `tidevec` wheel + sdist (`twine check --strict`) |
| `wheels-embedded` | `tidevec-embedded` wheels via cibuildwheel: manylinux_2_28 x86_64 and aarch64, macOS 14 arm64; CPython 3.9-3.13; each wheel runs `scripts/release_smoke_test.py --embedded` |
| `sdist-embedded` | `tidevec-embedded` sdist (headers vendored by `scripts/sync_embedded_headers.sh`) |
| `install-check` | fresh `pip install "tidevec[embedded]==X.Y.Z"` from the built files on Linux x86_64, Linux arm64 and macOS arm64 (Python 3.9 and 3.13), then the quickstart and smoke test, with timing |
| `docker-cpu` | per-arch image (native amd64 and arm64 runners), container smoke test, push by digest |
| `docker-cpu-manifest` | multi-arch tags `X.Y.Z`, `X.Y.Z-cpu`, and for stable releases `X.Y`, `X.Y-cpu`, `latest`, `cpu` |
| `docker-accel` | `X.Y.Z-gpu`/`-tpu`, `X.Y-gpu`/`-tpu`, `gpu`/`tpu` (amd64) |
| `publish-pypi` | `tidevec-embedded` first, then `tidevec` |
| `go-module` | `sdk/go/vX.Y.Z` tag (optional, see below) |
| `java` | deploy to GitHub Packages |
| `github-release` | GitHub release with CHANGELOG notes, Python dists and a C++ SDK tarball |

`workflow_dispatch` runs the build and test jobs without publishing; with the `testpypi` input it
uploads to TestPyPI.

## Secrets and settings to configure

| Name | Kind | Used for |
|---|---|---|
| `pypi` | environment | `publish-pypi`. Preferred: configure **PyPI trusted publishing** for both projects (`tidevec`, `tidevec-embedded`) with workflow `release.yml` and environment `pypi`; no token needed. |
| `PYPI_API_TOKEN` | secret (in `pypi`) | only if not using trusted publishing |
| `testpypi`, `TEST_PYPI_API_TOKEN` | environment, secret | optional TestPyPI dry runs |
| `DOCKER_USERNAME`, `DOCKER_PASSWORD` | repository secrets | Docker Hub pushes (`averm004/tidevec`) from `release.yml` and `docker.yml`; use an access token, not the account password |
| `dockerhub` | environment | gates the jobs that move Docker tags; add required reviewers if wanted |
| `CREATE_GO_TAG` | repository variable | `true` lets the workflow push the `sdk/go/vX.Y.Z` tag itself |
| `MAVEN_CENTRAL_ENABLED` | repository variable | placeholder; the Maven Central step is not implemented and fails if enabled |

### Go module

The Go module lives in `sdk/go`, so the Go proxy only sees versions tagged `sdk/go/vX.Y.Z`. Either set
`CREATE_GO_TAG=true`, or after the release run:

```bash
git tag sdk/go/v0.3.0 v0.3.0 && git push origin sdk/go/v0.3.0
```

### Maven Central

Not set up. Publishing needs a Sonatype Central account with the `io.tidevec` namespace verified, a
GPG signing key, and the `central-publishing-maven-plugin` in `sdk/java/pom.xml`. Until then Java
releases go to GitHub Packages only.

## Local dry run

```bash
uv build sdk/python --out-dir build-release/dist
scripts/sync_embedded_headers.sh && uv build sdk/python-embedded --out-dir build-release/dist
python -m venv /tmp/tv && /tmp/tv/bin/pip install --find-links build-release/dist "tidevec[embedded]==0.3.0"
/tmp/tv/bin/python examples/01_quickstart_embedded.py
/tmp/tv/bin/python scripts/release_smoke_test.py --embedded --expect-version 0.3.0
```
