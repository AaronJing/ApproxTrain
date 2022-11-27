
#ifndef MATMULAM_H_
#define MATMULAM_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "approx_mul_lut.h"

template <typename Device, typename T>
struct LaunchMatMul<Device, T>{
  void launch(
      const Device &d, const T& a, const T& b,
      const int batch, const int row_a, const int col_a, const int row_b,
      const int col_b, T* out,
      approx_mul_lut<Device>& mul_lut
      );
};
#if GOOGLE_CUDA 
template <typename T>
struct LaunchMatMul<Eigen::GpuDevice, T>{
  void launch(
      const Eigen::GpuDevice &d, const T& a, const T& b,
      const int batch, const int row_a, const int col_a, const int row_b,
      const int col_b, T* out,
      approx_mul_lut<Eigen::GpuDevice>& mul_lut
      );
};
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#endif  
