#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <vector>
#include <immintrin.h>
#include <iostream>

void scalarMatrixTranspose(std::vector<float>& A, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::swap(A[i * size + j], A[j * size + i]);
        }
    }
}

void avx256MatrixTranspose(std::vector<float>& A, int size) {
    std::vector<float> B(size * size);  

    for (int i = 0; i + 8 <= size; i += 8) {  
        for (int j = 0; j < size; ++j) {
            __m256 vecA = _mm256_loadu_ps(&A[i * size + j]); 
            _mm256_storeu_ps(&B[j * size + i], vecA);         
        }
    }
    
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            A[j * size + i] = B[i * size + j];  
        }
    }
}

TEST(MatrixTransposeTest, ScalarTranspose) {
    int size = 8;
    float k = 0;

    std::vector<float> A(size * size);
    for (int i = 0; i < size*size; i++) {
        A.push_back(k);
        k++;
    }

    std::vector<float> B = A;

    scalarMatrixTranspose(A, size);
    scalarMatrixTranspose(A, size);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            ASSERT_FLOAT_EQ(A[i * size + j], B[i * size + j]);
        }
    }
}

TEST(MatrixTransposeTest, AVX256Transpose) {
    int size = 8;
    float k = 0;

    std::vector<float> A(size * size);
    for (int i = 0; i < size*size; i++) {
        A.push_back(k);
        k++;
    }

    std::vector<float> B = A;

    avx256MatrixTranspose(A, size);
    avx256MatrixTranspose(A, size);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            ASSERT_FLOAT_EQ(A[i * size + j], B[i * size + j]);
        }
    }
}

static void BM_ScalarMatrixTranspose(benchmark::State& state) {
    int size = static_cast<int>(state.range(0));
    float k = 0;

    std::vector<float> A(size * size);
    for (int i = 0; i < size*size; i++) {
        A.push_back(k);
        k++;
    }

    for (auto _ : state) {
        scalarMatrixTranspose(A, size);
        benchmark::DoNotOptimize(A.data());
    }
}
BENCHMARK(BM_ScalarMatrixTranspose)->RangeMultiplier(2)->Range(1 << 5, 1 << 10)->Iterations(100)->Unit(benchmark::kMillisecond);

static void BM_AVX256MatrixTranspose(benchmark::State& state) {
    int size = static_cast<int>(state.range(0));
    float k = 0;

    std::vector<float> A(size * size);
    for (int i = 0; i < size*size; i++) {
        A.push_back(k);
        k++;
    }

    for (auto _ : state) {
        avx256MatrixTranspose(A, size);
        benchmark::DoNotOptimize(A.data());
    }
}
BENCHMARK(BM_AVX256MatrixTranspose)->RangeMultiplier(2)->Range(1 << 5, 1 << 10)->Iterations(100)->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int test_result = RUN_ALL_TESTS();

    std::cout << "\n=== Running Google Benchmark ===" << std::endl;
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return test_result;
    benchmark::RunSpecifiedBenchmarks();

    return test_result;
}
