cmake_minimum_required(VERSION 3.20)
project(tidevec_ann_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Threads REQUIRED)

set(TIDEVEC_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

# Same flags the main CMakeLists.txt applies to the header-only core.
add_executable(tvbench tvbench.cpp)
target_include_directories(tvbench PRIVATE ${TIDEVEC_INCLUDE})
target_compile_options(tvbench PRIVATE -O3 -ffast-math -march=native)
target_link_libraries(tvbench PRIVATE Threads::Threads)

# Same flags the tidevec-embedded wheel uses (portable: no -march=native,
# no -ffast-math), i.e. what `pip install tidevec[embedded]` users get.
add_executable(tvbench_portable tvbench.cpp)
target_include_directories(tvbench_portable PRIVATE ${TIDEVEC_INCLUDE})
target_compile_options(tvbench_portable PRIVATE -O3)
target_link_libraries(tvbench_portable PRIVATE Threads::Threads)
