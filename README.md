# Ramen

Ramen is a tiny library for using SIMD instructions to calculate mathematical functions efficiently.

## Requirement

- Intel CPU with supporting AVX2 and FMA
- Compiler with supporting C++17 or above

## Available Functions

Currently, three functions are available for double precision: `exp`, `sigmoid` and `tanh`.

## Usage

```c++
#include "ramen.hpp"

Ramen::Exp<8, 4>::call(double *dst, const double *src, int n);
Ramen::Sigmoid<8, 4>::call(double *dst, const double *src, int n);
Ramen::Tanh<8, 4>::call(double *dst, const double *src, int n);
```

## Benchmark

### Environment
- OS: macOS 10.15
- Processor: Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz
- Compiler: Apple clang version 11.0.3 (clang-1103.0.32.62)
- Compiler flags: `-O3 -ffast-math -mavx2 -mfma -fomit-frame-pointer`

### Result
```
// Comparison of elapsed time, maximum absolute (relative) error and root mean square (relative) error.
// Input: 1,000,000 generated random values based on standard normal distribution

[exp]
     naive(  c) 5.266581[ms]
     ramen(t01) 1.144960[ms](4.59979x faster than c)    mae=2.21511213335494519038e-16  rmse=5.10332480758338815603e-17
[sigmoid]
     naive(  c) 6.025321[ms]
     ramen(t01) 1.375231[ms](4.38132x faster than c)    mae=3.13699668049462979003e-16  rmse=4.02621468629261835387e-17
[tanh]
     naive(  c) 7.271433[ms]
     ramen(t01) 1.617250[ms](4.49617x faster than c)    mae=3.17437735752248113293e-10  rmse=3.23607653794666501049e-13
```
