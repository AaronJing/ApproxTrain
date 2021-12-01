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
            T* im2col, const int padding
          ); 
};

template <typename Device, typename T>
struct ConvamInputGradFunctor {
  void operator()(const Device& d, const T* grad, T* im2col, 
          const int hole_grad_width, const int hole_grad_height, 
          const int pad_top, const int pad_left, const T* filter, T* rsfilter,
          const int filter_rows, const int filter_cols, const int out_depth,
          const int stride_rows, const int stride_cols, const int batch,
          const int input_rows, const int input_cols, const int in_depth,
          T* output
          ); 
};

template <typename Device, typename T>
struct ConvamFilterGradFunctor{
  void operator()(const Device& d, const T* input, const T* grad, T* im2col,
          const int input_rows, const int input_cols, const int batch, 
          const int in_depth, const int out_cols, const int out_rows,
          const int out_depth, const int filter_left_offset, 
          const int filter_top_offset, const int stride_rows, 
          const int stride_cols, const int filter_cols, const int filter_rows, 
          T* output
          ); 
};
#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct ConvamFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, const T* input_data, 
            T* output_data, const int batch, const int out_rows, const out_cols, 
            const int out_depth, const int stride_cols, const int stride_rows,
            const int filter_left_offset, const int filter_top_offset,
            const int filter_rows, const int filter_cols, const int in_depth,
            const int input_cols, const int input_rows, const T* filter,
            T* im2col, const int padding
          ); 
};

template <typename T>
struct ConvamInputGradFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, const T* grad, T* im2col, 
          const int hole_grad_width, const int hole_grad_height, 
          const int pad_top, const int pad_left, const T* filter, T* rsfilter,
          const int filter_rows, const int filter_cols, const int out_depth,
          const int stride_rows, const int stride_cols, const int batch,
          const int input_rows, const int input_cols, const int in_depth,
          T* output
          ); 
};
template <typename T>
struct ConvamFilterGradFunctor<Eigen::GpuDevice, T>{
  void operator()(const Eigen::GpuDevice& d, const T* input, const T* grad, 
          T* im2col, const int input_rows, const int input_cols, 
          const int batch, const int in_depth, const int out_cols, 
          const int out_rows,const int out_depth, const int filter_left_offset, 
          const int filter_top_offset, const int stride_rows, 
          const int stride_cols, const int filter_cols, const int filter_rows, 
          T* output
          ); 
};
#endif

#endif CONVAM_H_
