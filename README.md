# Benchmark for Tridiagonal Solver

## How to Run the Benchmark

### The CUDA Version

Edit `Makefile` in the directory `Tridiagonal_CUDA` and `Makefile.inc` in the directory `Tridiagonal_CUDA/third_party/BPLib`, change CUDA Path and other options.

Change directory to `Tridiagonal_CUDA`, run `make` to compile.

Run `./test` with 5 arguments: precision, algorithm, matrix size, batch size, repetitions, with an optional argument: the output CSV file to append to.

### The AVX512 Version

Edit `Makefile` in the directory `Tridiagonal_AVX512`, change compiler and library Path and other options.

Change directory to `Tridiagonal_AVX512`, run `make` to compile.

Run `./test` with 5 arguments: precision, algorithm, matrix size, batch size, repetitions, with an optional argument: the output CSV file to append to.

