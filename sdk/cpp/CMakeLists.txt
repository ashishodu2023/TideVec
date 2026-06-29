# ================================================================
# TideVec C++ SDK
#
# Three install methods:
#   1. CMake FetchContent (easiest — no install needed)
#   2. vcpkg:  vcpkg install tidevec
#   3. Conan:  conan install tidevec/0.1.0
#   4. Manual: cmake --install build/
#
# Usage in your CMakeLists.txt (FetchContent):
#
#   include(FetchContent)
#   FetchContent_Declare(tidevec
#     GIT_REPOSITORY https://github.com/ashishodu2023/TideVec.git
#     GIT_TAG        v0.1.0
#     SOURCE_SUBDIR  sdk/cpp
#   )
#   FetchContent_MakeAvailable(tidevec)
#   target_link_libraries(my_app PRIVATE tidevec::tidevec)
#
# Usage in your code:
#
#   #include <tidevec/tidevec.hpp>
#
#   tidevec::TideVec db("localhost:6399");
#   db.create_collection("docs", {.dim=768, .half_life_ms=604800000});
#   db.upsert("docs", {{"id","v1"},{"embedding",vec}});
#   auto results = db.search("docs", query, {.top_k=10, .temporal_blend=0.3});
# ================================================================

cmake_minimum_required(VERSION 3.20)

project(tidevec-sdk
    VERSION      0.1.0
    DESCRIPTION  "TideVec C++ SDK — temporally-aware causal vector database"
    HOMEPAGE_URL "https://tidevec.com"
    LANGUAGES    CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Options ──────────────────────────────────────────────────────
option(TIDEVEC_SDK_BUILD_EXAMPLES "Build C++ SDK examples"    OFF)
option(TIDEVEC_SDK_BUILD_TESTS    "Build C++ SDK unit tests"  OFF)
option(TIDEVEC_SDK_INSTALL        "Generate install targets"  ON)

# ── Header-only interface library ────────────────────────────────
# The C++ SDK is header-only — all implementation in .hpp files
# No compilation required, links against the running TideVec server
add_library(tidevec INTERFACE)
add_library(tidevec::tidevec ALIAS tidevec)

target_include_directories(tidevec INTERFACE
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/../../include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(tidevec INTERFACE cxx_std_20)

# ── Dependencies (optional gRPC for remote connection) ───────────
find_package(Threads REQUIRED)
target_link_libraries(tidevec INTERFACE Threads::Threads)

# Try to find gRPC — falls back to HTTP-only client if not found
find_package(gRPC QUIET)
if(gRPC_FOUND)
    target_link_libraries(tidevec INTERFACE gRPC::grpc++)
    target_compile_definitions(tidevec INTERFACE TIDEVEC_GRPC_ENABLED)
    message(STATUS "TideVec SDK: gRPC found — gRPC transport enabled")
else()
    message(STATUS "TideVec SDK: gRPC not found — REST/HTTP transport only")
endif()

# ── Examples ─────────────────────────────────────────────────────
if(TIDEVEC_SDK_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ── Tests ────────────────────────────────────────────────────────
if(TIDEVEC_SDK_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Install / package ────────────────────────────────────────────
if(TIDEVEC_SDK_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    # Install headers
    install(
        DIRECTORY   ${CMAKE_SOURCE_DIR}/../../include/tidevec
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # Install CMake targets
    install(
        TARGETS  tidevec
        EXPORT   tidevecTargets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # Export targets
    install(
        EXPORT      tidevecTargets
        FILE        tidevecTargets.cmake
        NAMESPACE   tidevec::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tidevec
    )

    # Package config file
    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/tidevecConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/tidevecConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tidevec
    )

    # Version file
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/tidevecConfigVersion.cmake
        VERSION       ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/tidevecConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/tidevecConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tidevec
    )
endif()
