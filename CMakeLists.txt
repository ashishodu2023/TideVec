cmake_minimum_required(VERSION 3.20)
project(TideVec VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# Disable optional httplib features (no openssl/zlib/brotli deps)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Options ----------------------------------------------------
option(TIDEVEC_BUILD_TESTS "Build test binaries (set OFF for Docker builds)" ON)
option(TIDEVEC_CUDA        "Build with CUDA GPU support" OFF)
option(TIDEVEC_XLA         "Build with XLA/TPU support" OFF)
option(TIDEVEC_TLS         "Enable TLS support via OpenSSL" OFF)

# ---- Compiler flags ----------------------------------------------
set(TIDEVEC_WARN_FLAGS
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
)

# Common optimization flags across all architectures
set(TIDEVEC_OPT_FLAGS
        -O3
        -ffast-math
)

# Detect Architecture and Apply SIMD flags conditionally
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "Configuring for Apple Silicon / ARM64 architecture")
    # No -mavx2 or -mfma here; ARM64 uses native NEON instructions automatically via -O3
else()
    message(STATUS "Configuring for x86_64 architecture")
    # Release images must run on CPUs other than the build machine's:
    # docker/*/Dockerfile pass -DTIDEVEC_MARCH=x86-64-v3 (AVX2 + FMA).
    set(TIDEVEC_MARCH "native" CACHE STRING "x86_64 -march value (empty to omit)")
    if(TIDEVEC_MARCH)
        list(APPEND TIDEVEC_OPT_FLAGS -march=${TIDEVEC_MARCH})
    endif()
endif()

# ---- Interface library (header-only core) -----------------------
add_library(tidevec_headers INTERFACE)
target_include_directories(tidevec_headers INTERFACE
        ${CMAKE_SOURCE_DIR}/include
)
target_compile_options(tidevec_headers INTERFACE
        ${TIDEVEC_WARN_FLAGS}
        ${TIDEVEC_OPT_FLAGS}
)

# Enable threading (for shared_mutex, std::async, std::future)
find_package(Threads REQUIRED)

# ---- Server binary (always built) -------------------------------
add_executable(tidevec-server
        src/api/server_main.cpp
)
target_link_libraries(tidevec-server PRIVATE tidevec_headers Threads::Threads)
target_compile_options(tidevec-server PRIVATE ${TIDEVEC_WARN_FLAGS} -O2)

# Multi-Raft shard router (experimental; routes over several Raft groups)
add_executable(tidevec-router src/cluster/router_main.cpp)
target_link_libraries(tidevec-router PRIVATE tidevec_headers Threads::Threads)
target_compile_options(tidevec-router PRIVATE ${TIDEVEC_WARN_FLAGS} -O2)

if(TIDEVEC_TLS)
    # On tidevec_headers so every target sees the same httplib configuration.
    find_package(OpenSSL 3.0 REQUIRED)
    target_link_libraries(tidevec_headers INTERFACE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(tidevec_headers INTERFACE CPPHTTPLIB_OPENSSL_SUPPORT TIDEVEC_TLS)
    message(STATUS "TLS support ENABLED (OpenSSL ${OPENSSL_VERSION})")
endif()

# ---- CUDA (optional) --------------------------------------------
if(TIDEVEC_CUDA)
    enable_language(CUDA)
    # Keep CUDA at a broadly-supported dialect for nvcc toolchains
    # used by GitHub-hosted Docker builders.
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    add_library(tidevec_cuda STATIC
            src/accelerator/gpu_kernels.cu
    )
    set_target_properties(tidevec_cuda PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
    )
    target_link_libraries(tidevec_cuda
            PRIVATE CUDA::cudart CUDA::cublas
    )
    target_compile_definitions(tidevec_cuda PUBLIC TIDEVEC_CUDA_ENABLED)
    target_compile_options(tidevec_cuda PRIVATE
            -arch=sm_80
            --extended-lambda
            --use_fast_math
            -O3
    )
    target_link_libraries(tidevec-server PRIVATE tidevec_cuda CUDA::cudart CUDA::cublas)
    target_include_directories(tidevec-server PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(tidevec-server PRIVATE TIDEVEC_CUDA_ENABLED)
    message(STATUS "CUDA support ENABLED (sm_80)")
endif()

# ---- XLA/TPU (optional) -----------------------------------------
if(TIDEVEC_XLA)
    find_path(XLA_INCLUDE_DIR
        NAMES xla/client/xla_builder.h
        HINTS ${XLA_INCLUDE_DIR}
        PATHS
            /usr/local/include
            /usr/include
            $ENV{XLA_INCLUDE_DIR}
    )
    find_library(XLA_CLIENT_LIB
        NAMES xla_client
        HINTS ${XLA_LIB_DIR}
        PATHS
            /usr/local/lib
            /usr/lib
            $ENV{XLA_LIB_DIR}
    )
    if(NOT XLA_INCLUDE_DIR OR NOT XLA_CLIENT_LIB)
        message(FATAL_ERROR
            "TIDEVEC_XLA=ON but XLA client not found. "
            "Set XLA_INCLUDE_DIR and XLA_LIB_DIR, or build on a TensorFlow/XLA image.")
    endif()
    add_library(tidevec_tpu STATIC
            src/accelerator/tpu_kernels.cc
    )
    target_include_directories(tidevec_tpu PRIVATE ${XLA_INCLUDE_DIR})
    target_link_libraries(tidevec_tpu PRIVATE ${XLA_CLIENT_LIB})
    target_compile_definitions(tidevec_tpu PUBLIC TIDEVEC_XLA_ENABLED)
    target_link_libraries(tidevec-server PRIVATE tidevec_tpu)
    target_compile_definitions(tidevec-server PRIVATE TIDEVEC_XLA_ENABLED)
    message(STATUS "XLA/TPU support ENABLED")
endif()

# ---- Tests (skipped in Docker — cmake -DTIDEVEC_BUILD_TESTS=OFF) ----
if(TIDEVEC_BUILD_TESTS)
    message(STATUS "Building test targets")

    # Unit tests
    add_executable(tidevec_tests
            tests/unit/test_tidevec.cpp
    )
    target_link_libraries(tidevec_tests PRIVATE tidevec_headers Threads::Threads)
    target_compile_options(tidevec_tests PRIVATE
            ${TIDEVEC_WARN_FLAGS}
            ${TIDEVEC_OPT_FLAGS}
    )

    # Scale & Durability tests
    add_executable(tidevec_scale_tests
            tests/unit/test_scale_durability.cpp
    )
    target_link_libraries(tidevec_scale_tests PRIVATE tidevec_headers Threads::Threads)
    target_compile_options(tidevec_scale_tests PRIVATE
            ${TIDEVEC_WARN_FLAGS}
            ${TIDEVEC_OPT_FLAGS}
    )

    # API integration tests
    add_executable(tidevec_api_tests
            tests/integration/test_api.cpp
    )
    target_link_libraries(tidevec_api_tests PRIVATE tidevec_headers Threads::Threads)
    target_compile_options(tidevec_api_tests PRIVATE ${TIDEVEC_WARN_FLAGS} ${TIDEVEC_OPT_FLAGS})

    # Auth, tenancy, backup/restore, OTLP and TLS tests
    add_executable(tidevec_security_tests
            tests/integration/test_security.cpp
    )
    target_link_libraries(tidevec_security_tests PRIVATE tidevec_headers Threads::Threads)
    target_compile_options(tidevec_security_tests PRIVATE ${TIDEVEC_WARN_FLAGS} -O2)

    # C++ SDK against a live tidevec-server process. The SDK's types share
    # names with the server's, so this target must not see include/ core
    # headers (httplib alone is used for a stub server).
    add_executable(tidevec_cpp_sdk_tests
            tests/integration/test_cpp_sdk.cpp
    )
    target_include_directories(tidevec_cpp_sdk_tests PRIVATE
            ${CMAKE_SOURCE_DIR}/sdk/cpp/include ${CMAKE_SOURCE_DIR}/include/tidevec/api)
    target_link_libraries(tidevec_cpp_sdk_tests PRIVATE Threads::Threads)
    target_compile_options(tidevec_cpp_sdk_tests PRIVATE ${TIDEVEC_WARN_FLAGS} -O2)
    target_compile_definitions(tidevec_cpp_sdk_tests PRIVATE
            TIDEVEC_SERVER_BIN="$<TARGET_FILE:tidevec-server>")
    add_dependencies(tidevec_cpp_sdk_tests tidevec-server)

    # GPU/TPU Accelerator tests
    add_executable(tidevec_accel_tests
            tests/unit/test_accelerator.cpp
    )
    target_link_libraries(tidevec_accel_tests PRIVATE tidevec_headers Threads::Threads)
    target_compile_options(tidevec_accel_tests PRIVATE ${TIDEVEC_WARN_FLAGS} ${TIDEVEC_OPT_FLAGS})

    # Only add AVX/FMA on x86
    if(NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64"))
        target_compile_options(tidevec_accel_tests PRIVATE -mavx2 -mfma)
    endif()

    # Wire CUDA/XLA into accel tests if enabled
    if(TIDEVEC_CUDA)
        target_link_libraries(tidevec_accel_tests PRIVATE tidevec_cuda)
        target_compile_definitions(tidevec_accel_tests PUBLIC TIDEVEC_CUDA_ENABLED)
    endif()
    if(TIDEVEC_XLA)
        target_link_libraries(tidevec_accel_tests PRIVATE tidevec_tpu)
        target_compile_definitions(tidevec_accel_tests PUBLIC TIDEVEC_XLA_ENABLED)
    endif()

    # 11-nines durability tests (only if source file exists in repo)
    if(EXISTS ${CMAKE_SOURCE_DIR}/tests/unit/test_eleven_nines.cpp)
        add_executable(tidevec_eleven_nines_tests
                tests/unit/test_eleven_nines.cpp
        )
        target_link_libraries(tidevec_eleven_nines_tests PRIVATE tidevec_headers Threads::Threads)
        target_compile_options(tidevec_eleven_nines_tests PRIVATE
                ${TIDEVEC_WARN_FLAGS}
                ${TIDEVEC_OPT_FLAGS}
        )
        if(NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64"))
            target_compile_options(tidevec_eleven_nines_tests PRIVATE -mavx2 -mfma)
        endif()
    else()
        message(STATUS "Skipping tidevec_eleven_nines_tests (source not found)")
    endif()

    # Raft cluster tests (snapshots, membership, ReadIndex, pre-vote, faults)
    add_executable(tidevec_cluster_tests
            tests/cluster/test_raft_cluster.cpp
    )
    target_link_libraries(tidevec_cluster_tests PRIVATE tidevec_headers Threads::Threads)
    target_compile_options(tidevec_cluster_tests PRIVATE ${TIDEVEC_WARN_FLAGS} ${TIDEVEC_OPT_FLAGS})

    # 1B-scale benchmark
    add_executable(tidevec_bench_billion
            tests/benchmarks/bench_billion_scale.cpp
    )
    target_link_libraries(tidevec_bench_billion PRIVATE tidevec_headers Threads::Threads)
    target_compile_options(tidevec_bench_billion PRIVATE ${TIDEVEC_WARN_FLAGS} ${TIDEVEC_OPT_FLAGS})

    if(NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64" OR CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64"))
        target_compile_options(tidevec_bench_billion PRIVATE -mavx2 -mfma)
    endif()

    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(tidevec_bench_billion PRIVATE OpenMP::OpenMP_CXX)
    else()
        message(WARNING "OpenMP not found. Compiling tidevec_bench_billion without OpenMP.")
    endif()

else()
    message(STATUS "Skipping test targets (TIDEVEC_BUILD_TESTS=OFF)")
endif()

# ---- Install ----------------------------------------------------
install(TARGETS tidevec-server DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

# ---- Summary ----------------------------------------------------
message(STATUS "TideVec ${PROJECT_VERSION}")
message(STATUS "  Compiler   : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  C++ std    : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build dir  : ${CMAKE_BINARY_DIR}")
message(STATUS "  Build tests: ${TIDEVEC_BUILD_TESTS}")
message(STATUS "  CUDA       : ${TIDEVEC_CUDA}")
message(STATUS "  XLA/TPU    : ${TIDEVEC_XLA}")
