#ifndef GEMM_CUH
#define GEMM_CUH
template <typename T>
__global__ void gemm(size_t m, size_t n, size_t k,
   const T *a, size_t lda, const T *b, size_t ldb,
   T *c, size_t ldc, cudaTextureObject_t mant_lut,
   cudaTextureObject_t exp_lut, uint32_t* lut_mem
   );
#endif
