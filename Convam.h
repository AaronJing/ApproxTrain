// kernel_example.h
#ifndef CONVAM_H_
#define CONVAM_H_

#include <unsupported/Eigen/CXX11/Tensor>

template <typename Device, typename T>
struct ConvamFunctor {
  void operator()(const Device& d, const T* input_data, T* output_data
            const int batch, const int out_rows, const out_cols, 
            const int out_depth, const int stride_cols, const int stride_rows,
            const int filter_left_offset, const int filter_top_offset,
            const int filter_rows, const int filter_cols, const int in_depth,
            const int input_cols, const int input_rows, const T* filter,
            const T* im2col
          ); 
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct ConvamFunctor<Eigen::GpuDevice, T> {
  void operator()(const GPUDevice& d, const T* input_data, T* output_data,
            const int batch, const int out_rows, const out_cols, 
            const int out_depth, const int stride_cols, const int stride_rows,
            const int filter_left_offset, const int filter_top_offset,
            const int filter_rows, const int filter_cols, const int in_depth,
            const int input_cols, const int input_rows, const T* filter,
            const T* im2col
          ); 
};
#endif

#endif CONVAM_H_
