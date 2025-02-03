cmake_minimum_required(VERSION 3.14)

project(matrix-transpose)

set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED YES)
set(gtest_force_shared_crt on)

include(FetchContent)

set(BENCHMARK_ENABLE_TESTING off)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)

FetchContent_MakeAvailable(
    googletest
    googlebenchmark
)

add_executable(MatrixTranspose
  main.cpp
)

target_link_libraries(MatrixTranspose
  PRIVATE
    gtest_main
    benchmark::benchmark
    benchmark::benchmark_main
)

enable_testing()

include(GoogleTest)

gtest_discover_tests(MatrixTranspose)
