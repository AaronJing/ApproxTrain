#ifndef GEMM_CUH
#define GEMM_CUH

__global__ void gemm(size_t m, size_t n, size_t k,
   const float *a, size_t lda, const float *b, size_t ldb,
   float *c, size_t ldc);


#endif
