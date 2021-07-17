#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#ifndef GEMM_CUH
#define GEMM_CUH


#ifdef FMBM32_MULTIPLIER
   #define MULTIPLY(a,b) FPmultMBM_fast32((a),(b));
#else
   #ifdef FMBM16_MULTIPLIER
      #define MULTIPLY(a,b) FPmultMBM_fast16((a),(b));
   #else
      #define MULTIPLY(a,b) ((a)*(b));
   #endif
#endif

__global__ void gemm(size_t m, size_t n, size_t k,
   const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc);

__device__ float FPmultMBM_fast32(float Af, float Bf);

__device__ float FPmultMBM_fast16(float Af, float Bf);

#endif
