#include "Convam.h"
REGISTER_OP("Convam")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {float, int32}")
    .Attr("strides: list(int)")
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .SetShapeFn(::tensorflow::shape_inference::Conv2DShape);
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
template <typename Device, typename T>
class ConvamOpCPU : public OpKernel{
public:
  explicit ConvamOpCPU(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));
  }
  void Compute(OpKernelContext* context) override {
    //  grab input
    const Tensor& input = context->input(0);
    const Tensor& filter = context->input(1);
    // calculate parameters
    Conv2DDimensions dimensions;
    OP_REQUIRES_OK(context,
                   ComputeConv2DDimension(params_, input, filter, &dimensions));
    // get output shape
    TensorShape out_shape = ShapeFromFormat(
        params_.data_format, dimensions.batch, dimensions.out_rows,
        dimensions.out_cols, dimensions.out_depth);
    // allocate output tensor
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    // allocate im2col tensor for gpu
    Tensor im2col;
    int64 const oneinputsize = dimensions.input_rows * dimensions.input_cols * dimensions.in_depth;
    int64 const oneoutputsize = dimensions.out_rows* dimensions.out_cols * dimensions.out_depth;
    if(dimensions.filter_cols == 1 && dimensions.filter_rows == 1 && 
            dimensions.stride_rows == 1 && dimensions.stride_cols){
        size_t const block_size = 16;
        size_t const max_batch = (65536*block_size + 1 - block_size)/(dimensions.input_rows*dimensions.input_cols);
        if (dimensions.batch <= max_batch) {
            OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({dimensions.batch*dimensions.in_depth*dimensions.filter_rows*dimensions.filter_cols*dimensions.out_rows*dimensions.out_cols}), &im2col));
        } else {
            OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({max_batch*dimensions.in_depth*dimensions.filter_rows*dimensions.filter_cols*dimensions.out_rows*dimensions.out_cols}), &im2col));
        } 
    } else if(dimensions.filter_rows == dimensions.input_rows && dimensions.filter_cols == dimensions.input_cols&& params_.padding == 1){
        OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({dimensions.batch*dimensions.in_depth*dimensions.filter_rows*dimensions.filter_cols*dimensions.out_rows*dimensions.out_cols}), &im2col));
    } else {
        size_t const block_size = 16;
        size_t const max_batch = (65536*block_size + 1 - block_size)/(dimensions.out_rows*dimensions.out_cols);
        if (dimensions.batch <= max_batch) {
            OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({dimensions.batch*dimensions.out_cols*dimensions.out_rows, dimensions.in_depth*dimensions.filter_rows*dimensions.filter_cols}), &im2col));
        } else {
            OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({max_batch*dimensions.in_depth*dimensions.filter_rows*dimensions.filter_cols*dimensions.out_rows*dimensions.out_cols}), &im2col));
        }
    }
    auto output_data = output->flat<T>().data();
    auto input_data = input.flat<T>().data();
    auto filter_data = filter.flat<T>().data();
    auto im2col_data = im2col.flat<T>().data();
    // Calculate filter offset
    int filter_left_offset;
    int filter_top_offset;
    // VALID Padding
    if (params_.padding == 1) {
      filter_left_offset =
          ((dimensions.out_cols - 1) * dimensions.stride_cols + dimensions.filter_cols - dimensions.input_cols+1) / 2;
      filter_top_offset = ((dimensions.out_rows - 1) * dimensions.stride_rows + dimensions.filter_rows - dimensions.input_rows+1) /
          2;
      filter_left_offset = filter_left_offset>0?filter_left_offset:0;
      filter_top_offset = filter_top_offset>0?filter_top_offset:0;
    } else {
      filter_left_offset =
          ((dimensions.out_cols - 1) * dimensions.stride_cols + dimensions.filter_cols - dimensions.input_cols) / 2;
      filter_left_offset = filter_left_offset>0?filter_left_offset:0;
      filter_top_offset =
          ((dimensions.out_rows - 1) * dimensions.stride_rows + dimensions.filter_rows - dimensions.input_rows) /
          2;
      filter_top_offset = filter_top_offset>0?filter_top_offset:0;
    }
    ConvamFunctor<Device, T>()(
            context->eigen_device<Device>(),
            input_data,
            output_data,
            dimensions.batch,
            dimensions.out_rows,
            dimesnions.out_cols,
            dimensions.out_depth,
            dimensions.stride_cols,
            dimensions.stride_rows,
            filter_left_offset,
            filter_top_offset,
            dimensions.filter_rows,
            dimensions.filter_cols,
            dimensions.in_depth,
            dimensions.input_cols,
            dimensions.input_rows,
            filter_data,
            im2col_data
            );
  }
  private:
    Conv2DParameters params_;
  TF_DISALLOW_COPY_AND_ASSIGN(ConvamOpCPU);
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
// CPU specisialization for Convam filter grad
template <typename T>
struct ConvamFilterGradFunctor<CPUDevice, T>{
  void operator()(const CPUDevice& d, const T* input, const T* grad, T* im2col,
          const int input_rows, const int input_cols, const int batch, 
          const int in_depth, const int out_cols, const int out_rows,
          const int out_depth, const int filter_left_offset, 
          const int filter_top_offset, const int stride_rows, 
          const int stride_cols, const int filter_cols, const int filter_rows, 
          T* output
          ){

    for (int out_y = 0; out_y < filter_rows; ++out_y) {
        for (int out_x = 0; out_x < filter_cols; ++out_x) {
            for (int in_channel = 0; in_channel < in_depth; ++in_channel) {
                for (int out_channel = 0; out_channel < out_depth; ++out_channel) {
                    const int in_x_origin = (out_x * 1) - filter_left_offset;
                    const int in_y_origin = (out_y * 1) - filter_top_offset;
                    T total(0);
                    for (int grad_y = 0; grad_y < (out_rows-1)*stride_rows+1; ++grad_y) {
                        for (int grad_x = 0; grad_x < (out_cols-1)*stride_cols+1; ++grad_x) {
                            for (int g_batch = 0; g_batch < batch;
                                ++g_batch) {
                                const int in_x = in_x_origin + grad_x;
                                const int in_y = in_y_origin + grad_y;
                                T input_value;
                                if ((in_x >= 0) && (in_x < input_cols) && (in_y >= 0) &&
                                    (in_y < input_rows)) {
                                    input_value = in_data[
                                        (g_batch* input_rows * input_cols *
                                         input_channel) +
                                        (in_y * input_cols * input_channel) +
                                        (in_x * input_channel) + in_channel];
                                } else {
                                    input_value = T(0);
                                }
                  
                                const float i_y = (grad_y)/(float)stride_rows ;
                                const float i_x = (grad_x)/(float)stride_cols ;
                                const bool y = fmod(i_y,1)==float(0);
                                const bool x = fmod(i_x,1)==float(0);

                                const T grad_value = (x&y) ? 
                                    grad[(g_batch * out_cols * out_rows *
                                    grad_channel) +
                                    (int(i_y) * out_cols*grad_channel) +
                                    (int(i_x) * grad_channel) + out_channel]:0;
                                total += (input_value * grad_value);
                            }
                        }
                    }
                    output[(out_y * filter_cols * in_depth * 
                            out_depth) +
                        (out_x * in_depth *out_depth) +
                        (in_channel *out_depth) + out_channel] = total;

                }
            }
        }
    }
  } 
};
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

// CPU specisialization for ConvamInputGrad
template <typename T>
struct ConvamInputGradFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, const T* grad, T* im2col, 
          const int hole_grad_width, const int hole_grad_height, 
          const int pad_top, const int pad_left, const T* filter, T* rsfilter,
          const int filter_rows, const int filter_cols, const int out_depth,
          const int stride_rows, const int stride_cols, const int batch,
          const int input_rows, const int input_cols, const int in_depth,
          T* output, const int out_rows, const int out_cols
          ){
    for (int ibatch = 0; ibatch < batch; ++ibatch) {
        for (int out_y = 0; out_y < input_rows; ++out_y) {
            for (int out_x = 0; out_x < input_cols; ++out_x) {
                for (int in_channel = 0; in_channel < in_depth; ++in_channel) {
                    const int in_x_origin = out_x  - pad_left;
                    const int in_y_origin = out_y  - pad_top;  
                    float total(0);
                    for (int filter_y = 0; filter_y < filter_rows; ++filter_y) {
                        for (int filter_x = 0; filter_x < filter_cols; ++filter_x) {
                            for (int out_channel = 0; out_channel < out_depth;
                                ++out_channel) {
                                const int in_x = in_x_origin + filter_x;
                                const int in_y = in_y_origin + filter_y;
                                float input_value;
                                if((in_x >= 0) && (in_x < hole_grad_width) && (in_y >= 0) &&
                                (in_y < hole_grad_height)){
                                    const float i_y = (in_y)/(float)stride_rows ;
                                    const float i_x = (in_x)/(float)stride_cols ;
                                    const bool y = fmod(i_y,1)==float(0);
                                    const bool x = fmod(i_x,1)==float(0);
                                    input_value = (x&y) ?
                                    grad[(batch *grad_width*grad_height*out_depth) +
                                  (int(i_y) * grad_width*out_depth) +
                                  (int(i_x) * out_depth) + out_channel]:0;

                                } else {
                                    input_value = float(0);
                                }

                                const float filter_v = filter[((filter_rows-1-filter_y) * filter_cols * in_depth*out_depth) +
                                                    ((filter_cols-1-filter_x) * in_depth*out_depth) +
                                                    (in_channel * out_depth) + out_channel];

                                total += (input_value*filter_v);
                
                            }
                        }
                    }
                    output[(ibatch * input_cols * input_rows * in_depth) +
                        (out_y * input_cols * in_depth) +
                        (out_x *in_depth) + in_channel] = total;
                }
            }
        }
    }
  } 
};
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
