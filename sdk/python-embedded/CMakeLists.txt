cmake_minimum_required(VERSION 3.20)
project(tidevec_embedded LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)
find_package(Threads REQUIRED)

# Headers are vendored into ./include for sdists (scripts/sync_embedded_headers.sh);
# in a repo checkout they are read from the top-level include/ directory.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include/tidevec)
    set(TIDEVEC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
else()
    set(TIDEVEC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
endif()

pybind11_add_module(tidevec_native src/tidevec_native.cpp)
target_include_directories(tidevec_native PRIVATE ${TIDEVEC_INCLUDE_DIR})
target_link_libraries(tidevec_native PRIVATE Threads::Threads)
# No -march=native: wheels must run on any CPU of the target architecture.
target_compile_options(tidevec_native PRIVATE -O3)

install(TARGETS tidevec_native LIBRARY DESTINATION .)
