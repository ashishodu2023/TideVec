// ================================================================
// tidevec.hpp — TideVec C++ SDK
//
// Single-header public API for TideVec.
//
// Install via CMake FetchContent:
//   FetchContent_Declare(tidevec
//     GIT_REPOSITORY https://github.com/ashishodu2023/TideVec.git
//     GIT_TAG        v0.1.0
//     
//   )
//   FetchContent_MakeAvailable(tidevec)
//   target_link_libraries(my_app PRIVATE tidevec::tidevec)
//
// Or via vcpkg:
//   vcpkg install tidevec
//
// Usage:
//   #include <tidevec/tidevec.hpp>
//   tidevec::TideVec db("localhost:6399");
//   db.upsert("docs", {{"id","v1"},{"embedding",vec}});
//   auto hits = db.search("docs", query, {.top_k=10});
// ================================================================

#pragma once

#include <tidevec/client.hpp>
#include <tidevec/collection.hpp>
#include <tidevec/types.hpp>
#include <tidevec/half_life.hpp>
#include <tidevec/exceptions.hpp>
