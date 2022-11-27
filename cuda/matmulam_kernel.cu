
#define EIGEN_USE_GPU

#include "gpu_kernel_helper.h"
#include "error.cuh"
#include "denseam.h"
#include "approx_mul_lut.h"
#include "gemm.cuh"

using namespace tensorflow;
using GpuDevice = Eigen::GpuDevice;
template <typename T>
void LaunchMatMul<Eigen::GpuDevice, T>::launch(
      const Eigen::GpuDevice &d, const T& a, const T& b,
      const int batch, const int row_a, const int col_a, const int row_b,
      const int col_b, T* out,
      approx_mul_lut<Eigen::GpuDevice>& mul_lut
      ){
    const uint32_t mant_mask = mul_lut.get_mant_mask_();
    const uint8_t a_shift = mul_lut.get_a_shift_();
    const uint8_t b_shift = mul_lut.get_b_shift_();
    const uint8_t mant_bitwidth = mul_lut.get_mant_width_();
    const int m = row_a;
    const int n = col_b;
    const int k = col_a;
    const int lda = col_a;
    const int ldb = col_b;
    const int ldc = col_b;
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
    if (batch!=0){
        for(int i = 0; i < batch; i++){
            const T& temp_a = a[i*row_a*col_a];
            const T& temp_b = b[i*row_b*col_b];
            T& temp_c = out[i*row_a*col_b];
            gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, 
                    a, lda, b, ldb, out, ldc, 
                    mul_lut.get_mant_mul_lut_text_(), 
                    mant_mask, a_shift, b_shift, mant_bitwidth);    
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
        } 
    } else {
        gemm<T><<<gridSize, blockSize, 0, d.stream()>>>(m, n, k, 
                    a, lda, b, ldb, out, ldc, 
                    mul_lut.get_mant_mul_lut_text_(), 
                    mant_mask, a_shift, b_shift, mant_bitwidth);    
    
    }
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
