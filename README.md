# AVX256 Matrix Operations

This repository contains optimized implementations of matrix operations using AVX2 (AVX256) intrinsics. The two key operations included are:

- **Matrix Multiplication:** Performs square matrix multiplication using AVX2 instructions for improved performance.
- **Matrix Transpose:** Efficiently transposes a square matrix using AVX2 intrinsics.

## Features

- **SIMD Optimization:** Uses AVX2 intrinsics to process multiple floating-point values in parallel.
- **Performance Boost:** Optimized for high-speed computations on modern CPUs supporting AVX2.
- **Reusable Code Structure:** Both operations follow a similar build process.

---

## Getting Started

### Prerequisites

Ensure you have the following installed:

- **C++ Compiler:** GCC (4.7+), Clang (3.3+), or MSVC with AVX2 support.
- **CMake:** Version 3.10 or later.
- **CPU with AVX2 support.**

---

## Building the Project

### Steps to Build

1. Clone the repository:
   ```bash
   git clone https://github.com/Dhanshrepriya-148/SIMD-ASSIGNMENT
   cd (mat_mul/transpose)
   ```

2. Create a `build` directory and compile using CMake:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

3. Run the executables:
   - **For Matrix Multiplication:**
     ```bash
     ./MatrixMultiplication
     ```
   - **For Matrix Transpose:**
     ```bash
     ./MatrixTranspose 
     ```

