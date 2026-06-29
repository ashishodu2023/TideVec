// ================================================================
// cortexdb.hpp — CortexDB C++ SDK
//
// Single-header public API for CortexDB.
//
// Install via CMake FetchContent:
//   FetchContent_Declare(cortexdb
//     GIT_REPOSITORY https://github.com/ashishodu2023/Cortexdb.git
//     GIT_TAG        v0.1.0
//     SOURCE_SUBDIR  sdk/cpp
//   )
//   FetchContent_MakeAvailable(cortexdb)
//   target_link_libraries(my_app PRIVATE cortexdb::cortexdb)
//
// Or via vcpkg:
//   vcpkg install cortexdb
//
// Usage:
//   #include <cortexdb/cortexdb.hpp>
//   cortexdb::CortexDB db("localhost:6399");
//   db.upsert("docs", {{"id","v1"},{"embedding",vec}});
//   auto hits = db.search("docs", query, {.top_k=10});
// ================================================================

#pragma once

#include <cortexdb/client.hpp>
#include <cortexdb/collection.hpp>
#include <cortexdb/types.hpp>
#include <cortexdb/half_life.hpp>
#include <cortexdb/exceptions.hpp>
