#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <vector>
#include <immintrin.h>
#include <random>
#include <iostream>

void scalarMatrixMultiply(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            C[i * size + j] = 0;
            for (int k = 0; k < size; ++k) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

void avx256MatrixMultiply(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int size) {
    std::fill(C.begin(), C.end(), 0.0f);  

    for (int i = 0; i < size; i++) {  
        for (int j = 0; j < size; j += 8) {  
            __m256 vecC = _mm256_setzero_ps(); 
            for (int k = 0; k < size; ++k) {  
                __m256 vecA = _mm256_set1_ps(A[i * size + k]);  
                __m256 vecB = _mm256_set1_ps(B[k * size + j]);  
                vecC = _mm256_add_ps(vecC, _mm256_mul_ps(vecA, vecB)); 
            }

            _mm256_storeu_ps(&C[i * size + j], vecC);  
        }
    }
}
TEST(MatrixMultiplicationTest, ScalarMultiply) {
    int size = 8;
    std::vector<float> A(size * size, 1.0f);
    std::vector<float> B(size * size, 1.0f);
    std::vector<float> C(size * size);

    scalarMatrixMultiply(A, B, C, size);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            ASSERT_FLOAT_EQ(C[i * size + j], 8.0f);
        }
    }
}

TEST(MatrixMultiplicationTest, AVX256Multiply) {
    int size = 20;
    std::vector<float> A(size * size, 1.0f);
    std::vector<float> B(size * size, 1.0f);
    std::vector<float> C(size * size);

    avx256MatrixMultiply(A, B, C, size);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            ASSERT_FLOAT_EQ(C[i * size + j], 20.0f);
        }
    }
}

static void BM_ScalarMatrixMultiply(benchmark::State& state) {
    int size = static_cast<int>(state.range(0));
    std::vector<float> A(size * size, 1.0f);
    std::vector<float> B(size * size, 1.0f);
    std::vector<float> C(size * size);

    for (auto _ : state) {
        scalarMatrixMultiply(A, B, C, size);
        benchmark::DoNotOptimize(C.data());
    }
}
BENCHMARK(BM_ScalarMatrixMultiply)->RangeMultiplier(2)->Range(1 << 5, 1 << 10)->Iterations(10)->Unit(benchmark::kMillisecond);

static void BM_AVX256MatrixMultiply(benchmark::State& state) {
    int size = static_cast<int>(state.range(0));
    std::vector<float> A(size * size, 1.0f);
    std::vector<float> B(size * size, 1.0f);
    std::vector<float> C(size * size);

    for (auto _ : state) {
        avx256MatrixMultiply(A, B, C, size);
        benchmark::DoNotOptimize(C.data());
    }
}
BENCHMARK(BM_AVX256MatrixMultiply)->RangeMultiplier(2)->Range(1 << 5, 1 << 10)->Iterations(10)->Unit(benchmark::kMillisecond);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int test_result = RUN_ALL_TESTS();

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) return test_result;
    benchmark::RunSpecifiedBenchmarks();

    return test_result;
}
