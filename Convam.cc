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
                  const T filter_value =
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
            OP_REQUIRES_OK(context, context->allocate_temp(T, TensorShape({dimensions.batch*dimensions.in_depth*dimensions.filter_rows*dimensions.filter_cols*dimensions.out_rows*dimensions.out_cols}), &im2col));
        } else {
            OP_REQUIRES_OK(context, context->allocate_temp(T, TensorShape({max_batch*dimensions.in_depth*dimensions.filter_rows*dimensions.filter_cols*dimensions.out_rows*dimensions.out_cols}), &im2col));
        } 
    } else if(dimensions.filter_rows == dimensions.input_rows && dimensions.filter_cols == dimensions.input_cols&& params_.padding == 1){
        OP_REQUIRES_OK(context, context->allocate_temp(T, TensorShape({dimensions.batch*dimensions.in_depth*dimensions.filter_rows*dimensions.filter_cols*dimensions.out_rows*dimensions.out_cols}), &im2col));
    } else {
        size_t const block_size = 16;
        size_t const max_batch = (65536*block_size + 1 - block_size)/(dimensions.out_rows*dimensions.out_cols);
        if (dimensions.batch <= max_batch) {
            OP_REQUIRES_OK(context, context->allocate_temp(T, TensorShape({dimensions.batch*dimensions.out_cols*dimensions.out_rows, dimensions.in_depth*dimensions.filter_rows*dimensions.filter_cols}), &im2col));
        } else {
            OP_REQUIRES_OK(context, context->allocate_temp(T, TensorShape({max_batch*dimensions.in_depth*dimensions.filter_rows*dimensions.filter_cols*dimensions.out_rows*dimensions.out_cols}), &im2col));
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
template <typename Device, typename T>
REGISTER_OP("ConvamFilterGrad")
  .Attr("T: numbertype")
  .Input("input: T")
  .Input("filter_sizes: int32")
  .Input("out_backprop: T")
  .Output("grad_filter: T")
  .Attr("strides: list(int)")
  .Attr("dilations: list(int) = [1, 1, 1, 1]")
  .Attr(GetPaddingAttrString())
  .Attr(GetConvnetDataFormatAttrString())
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(1, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    });
class ConvamFilterGradOp: public OpKernel {
public:
  explicit ConvamFilterGradOp(OpKernelConstruction * context): OpKernel(context){
     string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    int stride_n = GetTensorDim(strides_, data_format_, 'N');
    int stride_c = GetTensorDim(strides_, data_format_, 'C');
    int stride_h = GetTensorDim(strides_, data_format_, 'H');
    int stride_w = GetTensorDim(strides_, data_format_, 'W');
    OP_REQUIRES(
        context, (stride_n == 1 && stride_c == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES(context, stride_h > 0 && stride_w > 0,
                errors::InvalidArgument(
                    "Row and column strides should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
    OP_REQUIRES(context, dilations_.size() == 4,
                errors::InvalidArgument("Sliding window dilations field must "
                                        "specify 4 dimensions"));
    int dilation_n = GetTensorDim(dilations_, data_format_, 'N');
    int dilation_c = GetTensorDim(dilations_, data_format_, 'C');
    int dilation_h = GetTensorDim(dilations_, data_format_, 'H');
    int dilation_w = GetTensorDim(dilations_, data_format_, 'W');
    OP_REQUIRES(context, (dilation_n == 1 && dilation_c == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilations in the batch and depth dimensions."));
    OP_REQUIRES(
        context, dilation_h > 0 && dilation_w > 0,
        errors::InvalidArgument("Dilated rates should be larger than 0."));

    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }
  void Compute(OpKernelContext* context) override{
        double begin = realtime();
    const Tensor& input = context->input(0);
    const Tensor& filter_sizes = context->input(1);
    const Tensor& out_backprop = context->input(2);

    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(filter_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropFilter: filter_sizes input must be 1-dim, not ",
            filter_sizes.dims()));
    TensorShape filter_shape;

    OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                filter_sizes.vec<int32>(), &filter_shape));
    Tensor* filter_backprop = nullptr;
    
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));
                       Tensor* output = nullptr;
    // If there is nothing to compute, return.
    if (filter_shape.num_elements() == 0) {
      return;
    }

    ConvBackpropDimensions dims;
    OP_REQUIRES_OK(context,
                   ConvBackpropComputeDimensions(
                       "Conv2DCustomBackpropFilter", /*num_spatial_dims=*/2,
                       input.shape(), filter_shape, out_backprop.shape(),
                       strides_, padding_, data_format_, &dims));
    int64 pad_top, pad_bottom;
    int64 pad_left, pad_right;
    OP_REQUIRES_OK(
        context,
        GetWindowedOutputSizeVerbose(
            dims.spatial_dims[0].input_size, dims.spatial_dims[0].filter_size,
            dims.spatial_dims[0].stride, padding_,
            &dims.spatial_dims[0].output_size, &pad_top, &pad_bottom));
    OP_REQUIRES_OK(
        context,
        GetWindowedOutputSizeVerbose(
            dims.spatial_dims[1].input_size, dims.spatial_dims[1].filter_size,
            dims.spatial_dims[1].stride, padding_,
            &dims.spatial_dims[1].output_size, &pad_left, &pad_right));

    const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
    const int stride_cols = GetTensorDim(strides_, data_format_, 'W');
    const int dilation_rows = GetTensorDim(dilations_, data_format_, 'H');
    const int dilation_cols = GetTensorDim(dilations_, data_format_, 'W');
    //get input dim
    const int input_batch = GetTensorDim(input,data_format_,'N');
    const int input_width = GetTensorDim(input,data_format_,'W');
    const int input_height = GetTensorDim(input,data_format_,'H');
    const int input_channel = GetTensorDim(input,data_format_,'C');
    //get grad dim
    const int grad_batch = GetTensorDim(out_backprop,data_format_,'N');
    const int grad_width = GetTensorDim(out_backprop,data_format_,'W');
    const int grad_height = GetTensorDim(out_backprop,data_format_,'H');
    const int grad_channel = GetTensorDim(out_backprop,data_format_,'C');
    //get filter dim
    const int filter_width = filter_shape.dim_size(1);
    const int filter_height = filter_shape.dim_size(0);
    const int filter_indepth = filter_shape.dim_size(2);
    const int filter_outdepth = filter_shape.dim_size(3);

    int64 filter_left_offset;
    int64 filter_top_offset;
    //VALID Padding
    if (padding_ == 1) {
      filter_top_offset =
          ((dims.output_size(0) - 1) * stride_rows + dims.filter_size(0) - dims.input_size(0) + 1) /
          2;
      filter_left_offset= ((dims.output_size(1) - 1) * stride_cols + dims.filter_size(1) -
                           dims.input_size(1) + 1) /                  
                          2;
      filter_left_offset = filter_left_offset>0?filter_left_offset:0;
      filter_top_offset = filter_top_offset>0?filter_top_offset:0;
    } else {
      filter_top_offset =
          ((dims.output_size(0) - 1) * stride_rows + dims.filter_size(0) - dims.input_size(0)) / 2;
      filter_left_offset=
          ((dims.output_size(1) - 1) * stride_cols + dims.filter_size(1) - dims.input_size(1)) /
          2;
      filter_left_offset = filter_left_offset>0?filter_left_offset:0;
      filter_top_offset = filter_top_offset>0?filter_top_offset:0;

    }
    Tensor im2col;
    OP_REQUIRES_OK(context, context->allocate_temp(T, 
    TensorShape({input_batch*input_channel*((grad_height-1)*stride_rows+1)*((grad_width-1)*stride_cols+1)*filter_width*filter_height})
    , &im2col));
    auto im2col_data = im2col.flat<T>().data();
    auto grad = out_backprop.flat<T>().data();
    auto in_data = input.flat<T>().data();
    auto out = filter_backprop->template flat<T>().data();
   
    ConvamFilterGradFunctor<Device, T>()(
            context->eigen_device<Device>(),
            in_data,
            grad,
            im2col_data,
            input_height,
            input_width,
            input_batch,
            input_channel,
            grad_width,
            grad_height,
            grad_channel,
            filter_left_offset,
            fitler_top_offset,
            stride_rows,
            stride_cols,
            filter_width,
            filter_height,
            out
            );
  }
  private:
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;
  TF_DISALLOW_COPY_AND_ASSIGN(ConvamFilterGradOpGPU);
 
};
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
                    T total(0);
                    for (int filter_y = 0; filter_y < filter_rows; ++filter_y) {
                        for (int filter_x = 0; filter_x < filter_cols; ++filter_x) {
                            for (int out_channel = 0; out_channel < out_depth;
                                ++out_channel) {
                                const int in_x = in_x_origin + filter_x;
                                const int in_y = in_y_origin + filter_y;
                                T input_value;
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
                                    input_value = T(0);
                                }

                                const T filter_v = filter[((filter_rows-1-filter_y) * filter_cols * in_depth*out_depth) +
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
REGISTER_OP("ConvamInputGrad")
    .Input("input_sizes: int32")
    .Attr("T: numbertype")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("grad_input: T")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    });
template <typename Device, typename T>
class ConvamInputGradOp public OpKernel {
    public:
     explicit ConvamInputGradOpCPU(OpKernelConstruction* context)
         : OpKernel(context) {
       string data_format;
       OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
       OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                   errors::InvalidArgument("Invalid data format"));
       OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
       OP_REQUIRES(context, strides_.size() == 4,
                   errors::InvalidArgument("Sliding window strides field must "
                                           "specify 4 dimensions"));
       int stride_n = GetTensorDim(strides_, data_format_, 'N');
       int stride_c = GetTensorDim(strides_, data_format_, 'C');
       int stride_h = GetTensorDim(strides_, data_format_, 'H');
       int stride_w = GetTensorDim(strides_, data_format_, 'W');
       OP_REQUIRES(
           context, (stride_n == 1 && stride_c == 1),
           errors::InvalidArgument("Current implementation does not yet support "
                                   "strides in the batch and depth dimensions."));
       OP_REQUIRES(context, stride_h > 0 && stride_w > 0,
                   errors::InvalidArgument(
                       "Row and column strides should be larger than 0."));
       OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
       OP_REQUIRES(context, dilations_.size() == 4,
                   errors::InvalidArgument("Sliding window dilations field must "
                                           "specify 4 dimensions"));
       int dilation_n = GetTensorDim(dilations_, data_format_, 'N');
       int dilation_c = GetTensorDim(dilations_, data_format_, 'C');
       int dilation_h = GetTensorDim(dilations_, data_format_, 'H');
       int dilation_w = GetTensorDim(dilations_, data_format_, 'W');
       OP_REQUIRES(context, (dilation_n == 1 && dilation_c == 1),
                   errors::InvalidArgument(
                       "Current implementation does not yet support "
                       "dilations in the batch and depth dimensions."));
       OP_REQUIRES(
           context, dilation_h > 0 && dilation_w > 0,
           errors::InvalidArgument("Dilated rates should be larger than 0."));
    
       OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
     }
    
     void Compute(OpKernelContext* context) override {
        const Tensor& input_sizes = context->input(0);
        const Tensor& filter = context->input(1);
        const Tensor& out_backprop = context->input(2);
        OP_REQUIRES(
            context, TensorShapeUtils::IsVector(input_sizes.shape()),
            errors::InvalidArgument(
                "Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
                input_sizes.dims()));
        TensorShape input_shape;
        OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
                                    input_sizes.vec<int32>(), &input_shape));
     
        Tensor* in_backprop = nullptr;
        OP_REQUIRES_OK(context,
                       context->allocate_output(0, input_shape, &in_backprop));
        
        if (input_shape.num_elements() == 0) {
          return;
        }
        ConvBackpropDimensions dims;
        OP_REQUIRES_OK(context,
                       ConvBackpropComputeDimensions(
                           "Conv2DCustomBackpropInput", /*num_spatial_dims=*/2,
                           input_shape, filter.shape(), out_backprop.shape(),
                           strides_, padding_, data_format_, &dims));
     
     
        const int filter_width = (int)dims.filter_size(1);
        const int filter_height = (int)dims.filter_size(0);
        const int output_channel = (int)dims.out_depth;
     
        const int input_batch = input_sizes.vec<int32>().data()[0];
        const int input_width = input_sizes.vec<int32>().data()[2];
        const int input_height = input_sizes.vec<int32>().data()[1];
        const int input_channel = input_sizes.vec<int32>().data()[3];
     
        //get grad dim
        const int grad_batch = GetTensorDim(out_backprop,data_format_,'N');
        const int grad_width = GetTensorDim(out_backprop,data_format_,'W');
        const int grad_height = GetTensorDim(out_backprop,data_format_,'H');
        const int grad_channel = GetTensorDim(out_backprop,data_format_,'C');
     
        const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
        const int stride_cols = GetTensorDim(strides_, data_format_, 'W');
        const int dilation_rows = GetTensorDim(dilations_, data_format_, 'H');
        const int dilation_cols = GetTensorDim(dilations_, data_format_, 'W');
     
        const int hole_grad_width = (grad_width-1)*stride_cols+1;
        const int hole_grad_height = (grad_height-1)*stride_rows+1;
     
     
        int64 forw_pad_top, forw_pad_bottom;
        int64 forw_pad_left, forw_pad_right;
        OP_REQUIRES_OK(
            context,
            GetWindowedOutputSizeVerbose(
                dims.spatial_dims[0].input_size, dims.spatial_dims[0].filter_size,
                dims.spatial_dims[0].stride, padding_,
                &dims.spatial_dims[0].output_size, &forw_pad_top, &forw_pad_bottom));
        OP_REQUIRES_OK(
            context,
            GetWindowedOutputSizeVerbose(
                dims.spatial_dims[1].input_size, dims.spatial_dims[1].filter_size,
                dims.spatial_dims[1].stride, padding_,
                &dims.spatial_dims[1].output_size, &forw_pad_left, &forw_pad_right));
        const int back_pad_top = filter_height - 1 - (int)forw_pad_top;
        const int back_pad_left = filter_width - 1 - (int)forw_pad_left;
     
        Tensor rsfilter;
        OP_REQUIRES_OK(context, context->allocate_temp(T, 
        TensorShape({output_channel*filter_width*filter_height*input_channel})
        , &rsfilter));
        auto grad = out_backprop.flat<T>().data();
        auto in_data = filter.flat<T>().data();
        auto out = in_backprop->template flat<T>().data();
        auto rsfilter_data = rsfilter.flat<T>().data();
     
        auto const oneinputsize = input_height*input_width*input_channel; 
        auto const oneoutputsize = grad_height*grad_width*output_channel; 
        size_t const block_size = 16;
        size_t const max_batch = (65536*block_size + 1 - block_size)/(input_height*input_width);
        Tensor im2col;
        if (input_batch <= max_batch) {
            OP_REQUIRES_OK(context, context->allocate_temp(T, 
            TensorShape({output_channel*filter_width*filter_height*input_height*input_width*grad_batch})
            , &im2col));
        } else {
            OP_REQUIRES_OK(context, context->allocate_temp(T, 
            TensorShape({output_channel*filter_width*filter_height*input_height*input_width*max_batch})
            , &im2col));
        }
        auto im2col_data = im2col.flat<T>().data();
        ConvamInputGradFunctor<Device, T>(
                context->eigen_device<Device>(),
                grad,
                im2col_data,
                hole_grad_width,
                hole_grad_height,
                back_pad_top,
                back_pad_left,
                in_data,
                rsfilter_data,
                filter_height,
                filter_width,
                output_channel,
                stride_rows,
                stride_cols,
                input_batch,
                input_height,
                input_width,
                input_channel,
                out
                );
     }
  private:
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;

  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(ConvamInputGradOpGPU);
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
