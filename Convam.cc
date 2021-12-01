#include "Convam.h"
// cpu specilisation
template <typename T>
struct ConvamFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, const T* input_data, T* output_data
            const int batch, const int out_rows, const out_cols, 
            const int out_depth, const int stride_cols, const int stride_rows,
            const int filter_left_offset, const int filter_top_offset,
            const int filter_rows, const int filter_cols, const int in_depth,
            const int input_cols, const int input_rows, const T* filter,
            const T* im2col
          ) {
    
    for (int batch_ = 0; batch_ < batch; ++batch_) {
      for (int out_y = 0; out_y < out_rows; ++out_y) {
        for (int out_x = 0; out_x < out_cols; ++out_x) {
          for (int out_channel = 0; out_channel < out_depth; ++out_channel) {
            const int in_x_origin = (out_x * stride_cols) - filter_left_offset;
            const int in_y_origin = (out_y * stride_rows) - filter_top_offset;
            T total(0);
            for (int filter_y = 0; filter_y < filter_rows; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_cols; ++filter_x) {
                for (int in_channel = 0; in_channel < in_depth;
                     ++in_channel) {
                  const int in_x = in_x_origin + filter_x;
                  const int in_y = in_y_origin + filter_y;
                  T input_value;
                  if ((in_x >= 0) && (in_x < input_cols) && (in_y >= 0) &&
                      (in_y < input_rows)) {
                    input_value =
                        input_data[(batch_ * input_cols * input_rows *
                                    in_depth) +
                                   (in_y * input_cols * in_depth) +
                                   (in_x * in_depth) + in_channel];
                  } else {
                    input_value = T(0);
                  }
                  const float filter_value =
                      filter_data[(filter_y * filter_cols * in_depth *
                                   out_depth) +
                                  (filter_x * in_depth * out_depth) +
                                  (in_channel * out_depth) + out_channel];
                  total += (input_value * filter_value);
                }
              }
            } 
            output_data[(batch_ * out_cols * out_rows * out_depth) +
                        (out_y * out_cols * out_depth) +
                        (out_x * out_depth) + out_channel] = total;
          }
        }
      }
    }
  }
};
// For forwardpropagation
// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Convam").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ConvamOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template class ConvamFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Convam").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ConvamOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA

// For backpropagation:filter
// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ConvamFilterGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ConvamFilterGradOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template class ConvamFilterGradFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Convam").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ConvamFilterGradOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA

// For backpropagation:input
// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ConvamInputGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ConvamInputGradOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template class ConvamInputGradFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ConvamInputGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ConvamInputGradOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
