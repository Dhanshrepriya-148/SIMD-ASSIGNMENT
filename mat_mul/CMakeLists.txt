cmake_minimum_required(VERSION 3.14)

project(matrix-multiplication)

# Set C++ standard to C++14
set(CMAKE_CXX_STANDARD 14)
set(CXX_STANDARD_REQUIRED YES)

# Optionally, force GoogleTest to use shared CRT
set(gtest_force_shared_crt ON)

# Include FetchContent to download dependencies
include(FetchContent)

# Disable testing in GoogleBenchmark
set(BENCHMARK_ENABLE_TESTING OFF)

# Fetch GoogleTest from its repository
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
)

# Fetch GoogleBenchmark from its repository
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)

# Make the content available
FetchContent_MakeAvailable(
    googletest
    googlebenchmark
)

# Add the executable for the main application
add_executable(MatrixMultiplicationMain
    main.cpp  # Your source file for the main application
)

# Link GoogleTest to the main executable
target_link_libraries(MatrixMultiplicationMain
    PRIVATE
        gtest_main
)

# Enable testing for the main executable
enable_testing()

# Discover and add tests for the main executable
include(GoogleTest)
gtest_discover_tests(MatrixMultiplicationMain TEST_DISCOVERY_TIMEOUT 30)

# Add the executable for the benchmarking application
add_executable(MatrixMultiplicationBenchmark
    benchmark.cpp  # Create a separate file for benchmarking code
)

# Link GoogleBenchmark to the benchmark executable
target_link_libraries(MatrixMultiplicationBenchmark
    PRIVATE
        benchmark::benchmark
        benchmark::benchmark_main
)

# Optionally, if you need to run benchmarks alongside tests:
# add_custom_target(run_benchmarks
#     COMMAND MatrixMultiplicationBenchmark
#     COMMENT "Running benchmarks"
# )
