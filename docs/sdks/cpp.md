# C++ SDK

A header-only C++20 REST client in
[`sdk/cpp`](https://github.com/ashishodu2023/TideVec/tree/main/sdk/cpp). It talks to a TideVec server;
to embed the engine itself in a C++ program, include the headers under `include/tidevec/` directly
(see `include/tidevec/core/collection.hpp`).

## Install with CMake FetchContent

```cmake
include(FetchContent)
FetchContent_Declare(tidevec
  GIT_REPOSITORY https://github.com/ashishodu2023/TideVec.git
  GIT_TAG        v0.3.0
  SOURCE_SUBDIR  sdk/cpp
)
FetchContent_MakeAvailable(tidevec)
target_link_libraries(my_app PRIVATE tidevec::tidevec)
```

The repository also contains a Conan recipe (`sdk/cpp/conanfile.py`) and a vcpkg manifest
(`sdk/cpp/vcpkg.json`), but the package is not in ConanCenter or the vcpkg registry; use them as local
recipes (`conan create sdk/cpp`). Each GitHub release also attaches a tarball of the SDK headers.

## Example

```cpp
#include <tidevec/tidevec.hpp>

tidevec::TideVec db("localhost:6399", {.api_key = "your-secret-key"});

db.create_collection({
    .name           = "docs",
    .dim            = 768,
    .half_life_ms   = tidevec::HalfLife::ONE_WEEK,
    .temporal_blend = 0.3f,
});

db.upsert("docs", {{
    .id        = "policy_v2",
    .embedding = embedding,
    .payload   = {{"source", "confluence"}},
    .edges     = {{.target_id = "policy_v1", .type = tidevec::EdgeType::UPDATES, .weight = 0.95f}},
}});
```

A complete program is in
[`sdk/cpp/examples/quickstart.cpp`](https://github.com/ashishodu2023/TideVec/blob/main/sdk/cpp/examples/quickstart.cpp).

Supersession is configured at creation (`.supersession = tidevec::SupersessionConfig{.match_field =
"doc_id"}`); search hits carry `superseded_by`, and `SearchOptions::superseded_penalty` ranks
superseded versions lower. `temporal_blend` is optional on both the collection and the search; when
unset, the server uses the collection's setting.

!!! note "Plain HTTP only"
    The C++ client has no TLS support and rejects `ClientConfig::tls = true`; reach a TLS endpoint
    through a local proxy. It has no delete-by-filter, scroll or alias calls; use the REST API (or
    the Python, Go or Java SDKs) for those.
