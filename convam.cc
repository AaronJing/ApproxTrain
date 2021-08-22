// #include "tensorflow/core/kernels/conv_ops.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/mirror_pad_mode.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/use_cudnn.h"
// #include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <sys/time.h>
#include "cuda/gpu_kernel_helper.h"
using namespace std;
using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;
static inline double realtime(void) {
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return tp.tv_sec + tp.tv_usec * 1e-6;
}
struct ConvBackpropSpatialDimension {
  int64 input_size;
  int64 filter_size;
  int64 output_size;
  int64 stride;
  int64 dilation;
  int64 expanded_output_size;

  // Number of padding elements to be added before/after this dimension of
  // the input when computing Conv?DBackpropInput.
  int64 pad_before, pad_after;
};

// Computed dimensions for a backwards convolution.
struct ConvBackpropDimensions {
  // Information about each spatial dimension.
  gtl::InlinedVector<ConvBackpropSpatialDimension, 3> spatial_dims;

  // Batch size.
  int64 batch_size;

  // Input and output feature depth.
  int64 in_depth, out_depth;

  // Convenience access methods for spatial dimensions properties.
  int64 input_size(int dim) const { return spatial_dims[dim].input_size; }
  int64 filter_size(int dim) const { return spatial_dims[dim].filter_size; }
  int64 output_size(int dim) const { return spatial_dims[dim].output_size; }
  int64 stride(int dim) const { return spatial_dims[dim].stride; }
  int64 dilation(int dim) const { return spatial_dims[dim].dilation; }

  // Compute padding for the given spatial dimension.
  int SpatialPadding(const Padding& padding, int dim) const;
};
int ConvBackpropDimensions::SpatialPadding(const Padding& padding,
                                           int dim) const {
  return (padding == VALID)
             ? 0
             : std::max<int>(
                   0, static_cast<int>((output_size(dim) - 1) * stride(dim) +
                                       (filter_size(dim) - 1) * dilation(dim) +
                                       1 - input_size(dim)));
}

Status ConvBackpropExtractAndVerifyDimensionV2(
    StringPiece label, const TensorShape& input_shape,
    const TensorShape& filter_shape, const TensorShape& output_shape,
    const gtl::ArraySlice<int32>& dilations, const std::vector<int32>& strides,
    Padding padding, int spatial_dim, int filter_spatial_dim,
    ConvBackpropSpatialDimension* dim) {
  dim->input_size = input_shape.dim_size(spatial_dim);
  dim->filter_size = filter_shape.dim_size(filter_spatial_dim);
  dim->output_size = output_shape.dim_size(spatial_dim);
  dim->stride = strides[spatial_dim];
  dim->dilation = dilations[spatial_dim];
  int64 out_size = 0, pad_size = 0;
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeV2(dim->input_size, dim->filter_size,
                                             dim->dilation, dim->stride,
                                             padding, &out_size, &pad_size));
  if (dim->output_size != out_size) {
    return errors::InvalidArgument(
        label, ": Size of out_backprop doesn't match computed: ", "actual = ",
        dim->output_size, ", computed = ", out_size,
        " spatial_dim: ", spatial_dim, " input: ", dim->input_size,
        " filter: ", dim->filter_size, " output: ", dim->output_size,
        " stride: ", dim->stride, " dilation: ", dim->dilation);
  }

  int64 effective_filter_size = (dim->filter_size - 1) * dim->dilation + 1;
  dim->expanded_output_size = (dim->output_size - 1) * dim->stride + 1;
  const auto padded_out_size = dim->input_size + effective_filter_size - 1;
  dim->pad_before = effective_filter_size - 1 - pad_size;
  dim->pad_after =
      padded_out_size - dim->expanded_output_size - dim->pad_before;
  VLOG(2) << label << ": expanded_out = " << dim->expanded_output_size
          << ", effective_filter_size = " << effective_filter_size
          << ", padded_out = " << padded_out_size
          << ", pad_before = " << dim->pad_before
          << ", pad_after = " << dim->pad_after
          << ", dilation = " << dim->dilation << ", strides = " << dim->stride;
  return Status::OK();
}

Status ConvBackpropExtractAndVerifyDimension(
    StringPiece label, const TensorShape& input_shape,
    const TensorShape& filter_shape, const TensorShape& output_shape,
    const std::vector<int32>& strides, Padding padding, int spatial_dim,
    int filter_spatial_dim, ConvBackpropSpatialDimension* dim) {
  static constexpr std::array<int32, 5> one_dilations = {{1, 1, 1, 1, 1}};
  return ConvBackpropExtractAndVerifyDimensionV2(
      label, input_shape, filter_shape, output_shape, one_dilations, strides,
      padding, spatial_dim, filter_spatial_dim, dim);
}

Status ConvBackpropComputeDimensionsV2(
    StringPiece label, int num_spatial_dims, const TensorShape& input_shape,
    const TensorShape& filter_shape, const TensorShape& out_backprop_shape,
    const gtl::ArraySlice<int32>& dilations, const std::vector<int32>& strides,
    Padding padding, TensorFormat data_format, ConvBackpropDimensions* dims) {
  // The + 2 in the following line is for the batch and feature dimensions.
  const int num_dims = num_spatial_dims + 2;
  if (input_shape.dims() != num_dims) {
    return errors::InvalidArgument(label, ": input must be ", num_dims,
                                   "-dimensional");
  }
  if (filter_shape.dims() != num_dims) {
    return errors::InvalidArgument(label, ": filter must be ", num_dims,
                                   "-dimensional");
  }
  if (out_backprop_shape.dims() != num_dims) {
    return errors::InvalidArgument(label, ": out_backprop must be ", num_dims,
                                   "-dimensional");
  }
  int batch_dim = GetTensorBatchDimIndex(num_dims, data_format);
  dims->batch_size = input_shape.dim_size(batch_dim);
  if (dims->batch_size != out_backprop_shape.dim_size(batch_dim)) {
    return errors::InvalidArgument(
        label, ": input and out_backprop must have the same batch size",
        "input batch: ", dims->batch_size,
        "outbackprop batch: ", out_backprop_shape.dim_size(batch_dim),
        " batch_dim: ", batch_dim);
  }

  int feature_dim = GetTensorFeatureDimIndex(num_dims, data_format);
  dims->in_depth = input_shape.dim_size(feature_dim);
  // The input and output feature dimensions are the second last and last
  // dimensions of the filter Tensor.
  VLOG(2) << "input vs filter_in depth " << dims->in_depth << " "
          << filter_shape.dim_size(num_dims - 2);
  if (dims->in_depth % filter_shape.dim_size(num_dims - 2)) {
    return errors::InvalidArgument(
        label, ": input depth must be evenly divisible by filter depth");
  }
  dims->out_depth = filter_shape.dim_size(num_dims - 1);
  if (dims->out_depth != out_backprop_shape.dim_size(feature_dim)) {
    return errors::InvalidArgument(
        label, ": filter and out_backprop must have the same out_depth");
  }
  dims->spatial_dims.resize(num_spatial_dims);
  for (int i = 0; i < num_spatial_dims; ++i) {
    int image_dim = GetTensorSpatialDimIndex(num_dims, data_format, i);
    TF_RETURN_IF_ERROR(ConvBackpropExtractAndVerifyDimensionV2(
        label, input_shape, filter_shape, out_backprop_shape, dilations,
        strides, padding, image_dim, i, &dims->spatial_dims[i]));
  }
  return Status::OK();
}

Status ConvBackpropComputeDimensions(StringPiece label, int num_spatial_dims,
                                     const TensorShape& input_shape,
                                     const TensorShape& filter_shape,
                                     const TensorShape& out_backprop_shape,
                                     const std::vector<int32>& strides,
                                     Padding padding, TensorFormat data_format,
                                     ConvBackpropDimensions* dims) {
  static constexpr std::array<int32, 5> one_dilations = {{1, 1, 1, 1, 1}};
  return ConvBackpropComputeDimensionsV2(
      label, num_spatial_dims, input_shape, filter_shape, out_backprop_shape,
      one_dilations, strides, padding, data_format, dims);
}

// Common code between implementations of Conv?DBackpropInput and
// Conv?DBackpropFilter. Verifies that the dimensions all match, and computes
// sizes/padding for the spatial dimensions.
Status ConvBackpropComputeDimensions(StringPiece label, int num_spatial_dims,
                                     const TensorShape& input_shape,
                                     const TensorShape& filter_shape,
                                     const TensorShape& out_backprop_shape,
                                     const std::vector<int32>& strides,
                                     Padding padding, TensorFormat data_format,
                                     ConvBackpropDimensions* dims);

// The V2 version computes the same outputs with arbitrary dilation rate.
// TODO(b/67112639): Merge V2 versions and the original versions eventually.
Status ConvBackpropComputeDimensionsV2(
    StringPiece label, int num_spatial_dims, const TensorShape& input_shape,
    const TensorShape& filter_shape, const TensorShape& out_backprop_shape,
    const gtl::ArraySlice<int32>& dilations, const std::vector<int32>& strides,
    Padding padding, TensorFormat data_format, ConvBackpropDimensions* dims);
  // namespace tensorflow

// this implementation does not support padding and diltation currently, please padding before using conv
// Convolution parameters specified by Op attributes.
struct Conv2DParameters {
  std::vector<int32> dilations;
  std::vector<int32> strides;
  Padding padding;
  TensorFormat data_format;
};

// Convolution dimensions inferred from parameters, input and filter tensors.
struct Conv2DDimensions {
  int batch;
  int input_rows;
  int input_cols;
  int in_depth;

  int filter_rows;
  int filter_cols;
  int patch_depth;
  int out_depth;

  int stride_rows;
  int stride_cols;

  int dilation_rows;
  int dilation_cols;

  int64 out_rows;
  int64 out_cols;
  int64 pad_rows;
  int64 pad_cols;
};

// Initializes and validates Conv2D parameters configured by OpKernel
// attributes.
Status InitConv2DParameters(const OpKernelConstruction* context,
                            Conv2DParameters* params);

// Computes and validates convolutions dimensions from Conv2D parameters. If
// parameters are valid, dimensions will be updated with derived convolution
// dimensions, otherwise error will be returned.
Status ComputeConv2DDimension(const Conv2DParameters& params,
                              const Tensor& input, const Tensor& filter,
                              Conv2DDimensions* dimensions);

#define TF_REQUIRES(EXP, STATUS)                \
  do {                                          \
    if (!TF_PREDICT_TRUE(EXP)) return (STATUS); \
  } while (false)

Status InitConv2DParameters(const OpKernelConstruction* context,
                            Conv2DParameters* params) {
  TF_RETURN_IF_ERROR(context->GetAttr("dilations", &params->dilations));
  TF_RETURN_IF_ERROR(context->GetAttr("strides", &params->strides));
  TF_RETURN_IF_ERROR(context->GetAttr("padding", &params->padding));
  string data_format_string;
  TF_RETURN_IF_ERROR(context->GetAttr("data_format", &data_format_string));
  TF_REQUIRES(FormatFromString(data_format_string, &params->data_format),
              errors::InvalidArgument("Invalid data format"));

  const auto& strides = params->strides;
  const auto& dilations = params->dilations;
  const auto& data_format = params->data_format;

  TF_REQUIRES(dilations.size() == 4,
              errors::InvalidArgument("Sliding window dilations field must "
                                      "specify 4 dimensions"));
  TF_REQUIRES(strides.size() == 4,
              errors::InvalidArgument("Sliding window strides field must "
                                      "specify 4 dimensions"));
  const int64 stride_n = GetTensorDim(strides, data_format, 'N');
  const int64 stride_c = GetTensorDim(strides, data_format, 'C');
  const int64 stride_h = GetTensorDim(strides, data_format, 'H');
  const int64 stride_w = GetTensorDim(strides, data_format, 'W');
  TF_REQUIRES(
      stride_n == 1 && stride_c == 1,
      errors::InvalidArgument("Current implementation does not yet support "
                              "strides in the batch and depth dimensions."));
  TF_REQUIRES(stride_h > 0 && stride_w > 0,
              errors::InvalidArgument(
                  "Row and column strides should be larger than 0."));

  const int64 dilation_n = GetTensorDim(dilations, data_format, 'N');
  const int64 dilation_c = GetTensorDim(dilations, data_format, 'C');
  const int64 dilation_h = GetTensorDim(dilations, data_format, 'H');
  const int64 dilation_w = GetTensorDim(dilations, data_format, 'W');
  TF_REQUIRES(
      dilation_n == 1 && dilation_c == 1,
      errors::InvalidArgument("Current implementation does not yet support "
                              "dilations in the batch and depth dimensions."));
  TF_REQUIRES(
      dilation_h > 0 && dilation_w > 0,
      errors::InvalidArgument("Dilated rates should be larger than 0."));

  return Status::OK();
}

Status ComputeConv2DDimension(const Conv2DParameters& params,
                              const Tensor& input, const Tensor& filter,
                              Conv2DDimensions* dimensions) {
  // Check that 2D convolution input and filter have exactly 4 dimensions.
  TF_REQUIRES(input.dims() == 4,
              errors::InvalidArgument("input must be 4-dimensional",
                                      input.shape().DebugString()));
  TF_REQUIRES(filter.dims() == 4,
              errors::InvalidArgument("filter must be 4-dimensional: ",
                                      filter.shape().DebugString()));
  for (int i = 0; i < 3; i++) {
    TF_REQUIRES(
        FastBoundsCheck(filter.dim_size(i), std::numeric_limits<int>::max()),
        errors::InvalidArgument("filter too large"));
  }

  // The last dimension for input is in_depth. Check that it is the same as the
  // filter's in_depth or it is evenly divisible by filter's in_depth.
  const int64 in_depth_raw = GetTensorDim(input, params.data_format, 'C');
  const int64 patch_depth_raw = filter.dim_size(2);
  TF_REQUIRES(FastBoundsCheck(in_depth_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Input depth too large"));
  TF_REQUIRES(FastBoundsCheck(patch_depth_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Patch depth too large"));
  const int in_depth = static_cast<int>(in_depth_raw);
  const int patch_depth = static_cast<int>(patch_depth_raw);
  TF_REQUIRES(in_depth % patch_depth == 0,
              errors::InvalidArgument(
                  "input depth must be evenly divisible by filter depth: ",
                  in_depth, " vs ", patch_depth));

  // The last dimension for filter is out_depth.
  const int out_depth = static_cast<int>(filter.dim_size(3));

  // The second dimension for input is rows/height.
  // The first dimension for filter is rows/height.
  const int64 input_rows_raw = GetTensorDim(input, params.data_format, 'H');
  TF_REQUIRES(FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Input rows too large"));
  const int input_rows = static_cast<int>(input_rows_raw);
  const int filter_rows = static_cast<int>(filter.dim_size(0));

  // The third dimension for input is columns/width.
  // The second dimension for filter is columns/width.
  const int64 input_cols_raw = GetTensorDim(input, params.data_format, 'W');
  TF_REQUIRES(FastBoundsCheck(input_cols_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("Input cols too large"));
  const int input_cols = static_cast<int>(input_cols_raw);
  const int filter_cols = static_cast<int>(filter.dim_size(1));

  // The first dimension for input is batch.
  const int64 batch_raw = GetTensorDim(input, params.data_format, 'N');
  // printf("batch_raw %d\n",(int)batch_raw);
  TF_REQUIRES(FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
              errors::InvalidArgument("batch is too large"));
  const int batch = static_cast<int>(batch_raw);

  // Take the stride and dilation from the second and third dimensions only (we
  // do not support striding or dilation on the batch or depth dimension).
  const int stride_rows = GetTensorDim(params.strides, params.data_format, 'H');
  const int stride_cols = GetTensorDim(params.strides, params.data_format, 'W');
  const int dilation_rows =
      GetTensorDim(params.dilations, params.data_format, 'H');
  const int dilation_cols =
      GetTensorDim(params.dilations, params.data_format, 'W');

  // Compute windowed output sizes for rows and columns.
  int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeV2(
      input_rows, filter_rows, dilation_rows, stride_rows, params.padding,
      &out_rows, &pad_rows));
  TF_RETURN_IF_ERROR(GetWindowedOutputSizeV2(
      input_cols, filter_cols, dilation_cols, stride_cols, params.padding,
      &out_cols, &pad_cols));

  dimensions->batch = batch;
  dimensions->input_rows = input_rows;
  dimensions->input_cols = input_cols;
  dimensions->in_depth = in_depth;
  dimensions->filter_rows = filter_rows;
  dimensions->filter_cols = filter_cols;
  dimensions->patch_depth = patch_depth;
  dimensions->out_depth = out_depth;
  dimensions->stride_rows = stride_rows;
  dimensions->stride_cols = stride_cols;
  dimensions->dilation_rows = dilation_rows;
  dimensions->dilation_cols = dilation_cols;
  dimensions->out_rows = out_rows;
  dimensions->out_cols = out_cols;
  dimensions->pad_rows = pad_rows;
  dimensions->pad_cols = pad_cols;

  return Status::OK();
}

#undef TF_REQUIRES

// REGISTER_OP("Convam")
//     .Input("input: float")
//     .Input("filter: float")
//     .Output("output: float")
//     .Attr("strides: list(int)")
//     .Attr("use_cudnn_on_gpu: bool = true")
//     .Attr(padding::GetPaddingAttrStringWithExplicit())
//     .Attr(GetExplicitPaddingsAttrString())
//     .Attr(GetConvnetDataFormatAttrString())
//     .Attr("dilations: list(int) = [1, 1, 1, 1]")
//     .SetShapeFn(::tensorflow::shape_inference::Conv2DShapeWithExplicitPadding);

REGISTER_OP("Convam")
    .Input("input: float")
    .Input("filter: float")
    .Output("output: float")
    .Attr("strides: list(int)")
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .SetShapeFn(::tensorflow::shape_inference::Conv2DShape);


class ConvamOpCPU : public OpKernel{
public:
  explicit ConvamOpCPU(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));
  }
  void Compute(OpKernelContext* context) override {
    // ----------test for shape_inference-----------
    // TensorShape output_shape;
    // output_shape.AddDim(2);
    // output_shape.AddDim(2);
    // Tensor* output = NULL;
    // OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    // auto output_tensor = output->matrix<float>();
    // output_tensor(0,0) = 1;
    // output_tensor(0,1) = 2;
    // output_tensor(1,0) = 3;
    // output_tensor(1,1) = 4;
    // ----------test for shape_inference-----------

    // ----------test for output shape-----------
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter = context->input(1);
    Conv2DDimensions dimensions;
    OP_REQUIRES_OK(context,
                   ComputeConv2DDimension(params_, input, filter, &dimensions));

    TensorShape out_shape = ShapeFromFormat(
        params_.data_format, dimensions.batch, dimensions.out_rows,
        dimensions.out_cols, dimensions.out_depth);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    // ----------test for output shape-----------
    
    // ----------grab shape------------
    // const int input_batch = input.shape().dim_size(0);
    // const int input_height = input.shape().dim_size(1);
    // const int input_width = input.shape().dim_size(2);
    // const int input_channel = input.shape().dim_size(3);

    // const int filter_height = filter.shape().dim_size(0);
    // const int filter_width = filter.shape().dim_size(1);
    // const int filter_in = filter.shape().dim_size(2);
    // const int filter_out = filter.shape().dim_size(3);

    // ----------grab shape------------
    
    // ----------test for accessing tensor -----------
    auto output_data = output->flat<float>().data();
    auto input_data = input.flat<float>().data();
    auto filter_data = filter.flat<float>().data();
    // for(int i = 0; i < input_batch*input_height*input_width*input_channel; i++)
    //   printf("%f\n\n\n",in_put[i]);
    //printf("padding type%d\n",params_.padding);
    // ----------test for accessing tensor -----------

    // ----------test for forward propagation-----------
    int filter_left_offset;
    int filter_top_offset;
    // VALID Padding
    if (params_.padding == 1) {
      filter_left_offset =
          ((dimensions.out_cols - 1) * dimensions.stride_cols + dimensions.filter_cols - dimensions.input_cols+1) / 2;
      // printf("filter_left_offset = %d, out_rows = %d, stride_cols = %d, filter_rows = %d, input_rows = %d\n",
      //             filter_left_offset,(int)dimensions.out_rows,dimensions.stride_cols,dimensions.filter_rows,dimensions.input_rows);
      filter_top_offset = ((dimensions.out_rows - 1) * dimensions.stride_rows + dimensions.filter_rows - dimensions.input_rows+1) /
          2;
      filter_left_offset = filter_left_offset>0?filter_left_offset:0;
      filter_top_offset = filter_top_offset>0?filter_top_offset:0;
      // printf("filter_top_offset = %d, out_cols = %d, stride_rows = %d, filter_cols = %d, input_cols = %d\n",
      //             filter_top_offset,(int)dimensions.out_cols,dimensions.stride_rows,dimensions.filter_cols,dimensions.input_cols);
    } else {
      filter_left_offset =
          ((dimensions.out_cols - 1) * dimensions.stride_cols + dimensions.filter_cols - dimensions.input_cols) / 2;
      // printf("filter_left_offset = %d, out_rows = %d, stride_cols = %d, filter_rows = %d, input_rows = %d\n",
      //             filter_left_offset,(int)dimensions.out_rows,dimensions.stride_cols,dimensions.filter_rows,dimensions.input_rows);
      filter_left_offset = filter_left_offset>0?filter_left_offset:0;
      filter_top_offset =
          ((dimensions.out_rows - 1) * dimensions.stride_rows + dimensions.filter_rows - dimensions.input_rows) /
          2;
      filter_top_offset = filter_top_offset>0?filter_top_offset:0;
      // printf("filter_top_offset = %d, out_cols = %d, stride_rows = %d, filter_cols = %d, input_cols = %d\n",
      //             filter_top_offset,(int)dimensions.out_cols,dimensions.stride_rows,dimensions.filter_cols,dimensions.input_cols);
    }
    
    // If we've got multiple images in our input, work through each of them.
    for (int batch = 0; batch < dimensions.batch; ++batch) {
      // Walk through all the output image values, sliding the filter to
      // different positions in the input.
      for (int out_y = 0; out_y < dimensions.out_rows; ++out_y) {
        for (int out_x = 0; out_x < dimensions.out_cols; ++out_x) {
          // Each filter kernel produces one output channel.
          for (int out_channel = 0; out_channel < dimensions.out_depth; ++out_channel) {
            // We're going to calculate a single output value, which means we
            // need to multiply a three dimensional kernel of weights against
            // the current location within the input image.
            /*
             *-------------------------------...
             |\ ^
             | \in_depth
             |  \ v
             |   *-------------------------------...
             |   |            ^
             |   |       in_y_origin
             |   |            v   \
             |   |<in_x_origin>*---*^
             |   |            \|   |dimensions.filter_cols
             .   |             *---*v
             .   |             <--->
             .         dimensions.filter_rows
             .
            */
            const int in_x_origin = (out_x * dimensions.stride_cols) - filter_left_offset;
            const int in_y_origin = (out_y * dimensions.stride_rows) - filter_top_offset;
            float total(0);
            for (int filter_y = 0; filter_y < dimensions.filter_rows; ++filter_y) {
              for (int filter_x = 0; filter_x < dimensions.filter_cols; ++filter_x) {
                for (int in_channel = 0; in_channel < dimensions.in_depth;
                     ++in_channel) {
                  const int in_x = in_x_origin + filter_x;
                  const int in_y = in_y_origin + filter_y;
                  // printf("%d in_x forward\n",in_x);
                  // printf("%d in_y\n",in_y);
                  float input_value;
                  // If the location is outside the bounds of the input image,
                  // use zero as a default value.
                  if ((in_x >= 0) && (in_x < dimensions.input_cols) && (in_y >= 0) &&
                      (in_y < dimensions.input_rows)) {
                    input_value =
                        input_data[(batch * dimensions.input_cols * dimensions.input_rows *
                                    dimensions.in_depth) +
                                   (in_y * dimensions.input_cols * dimensions.in_depth) +
                                   (in_x * dimensions.in_depth) + in_channel];
                  } else {
                    input_value = float(0);
                  }
                  const float filter_value =
                      filter_data[(filter_y * dimensions.filter_cols * dimensions.in_depth *
                                   dimensions.out_depth) +
                                  (filter_x * dimensions.in_depth * dimensions.out_depth) +
                                  (in_channel * dimensions.out_depth) + out_channel];
                  total += (input_value * filter_value);
                }
              }
            } 
           // printf("\n\n\n");
          //  printf("%f\n",total);
            output_data[(batch * dimensions.out_cols * dimensions.out_rows * dimensions.out_depth) +
                        (out_y * dimensions.out_cols * dimensions.out_depth) +
                        (out_x * dimensions.out_depth) + out_channel] = total;
          }
        }
      }
    }
    // std::fstream fs;
    // fs.open("cpu_log", std::fstream::in | std::fstream::out | std::fstream::app);
    // // ----------test for forward propagation-----------
    // for(int i = 0; i < dimensions.batch; i++){
    //   for(int j = 0; j < dimensions.out_rows; j++){
    //     for(int k = 0; k < dimensions.out_cols; k++){
    //       for(int l = 0; l < dimensions.out_depth; l++){
    //           // printf("(%d, %d, %d, %d) %f\n",i,j,k,l,output_data[(i * dimensions.out_cols * dimensions.out_rows * dimensions.out_depth) +
    //           //           (j * dimensions.out_cols * dimensions.out_depth) +
    //           //           (k * dimensions.out_depth) + l]);
    //           fs << i <<", " << j << ", " <<k<<", " << l << ",  " << output_data[(i * dimensions.out_cols * dimensions.out_rows * dimensions.out_depth) +
    //                     (j * dimensions.out_cols * dimensions.out_depth) +
    //                     (k * dimensions.out_depth) + l]<<"\n";
              
    //       }
    //     }
    //   }
    // }
    // // printf("From CPU\n");
    // fs << "From CPU\n";
    // fs.close();
  }
  
  private:
    Conv2DParameters params_;
  TF_DISALLOW_COPY_AND_ASSIGN(ConvamOpCPU);
};

REGISTER_KERNEL_BUILDER(Name("Convam").Device(DEVICE_CPU), ConvamOpCPU);



void ConvamKernellLauncher(
        const GPUDevice &d,
  const float* inputs,
  const float* filter,
  float* im2col,
  const int batch,
  const int in_row,
  const int in_col,
  const int out_row,
  const int out_col,
  const int out_depth,
  const int in_depth,
  const int filter_row,
  const int filter_col,
  const int stride_row,
  const int stride_col,
  // Padding
  const int left_offset,
  const int top_offset,
  const int padding,
  float* output
);

class ConvamOpGPU : public OpKernel{
public:
  explicit ConvamOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, InitConv2DParameters(context, &params_));
  }
  void Compute(OpKernelContext* context) override {
    double begin = realtime();
    const Tensor& input = context->input(0);
    const Tensor& filter = context->input(1);
    Conv2DDimensions dimensions;
    OP_REQUIRES_OK(context,
                   ComputeConv2DDimension(params_, input, filter, &dimensions));

    TensorShape out_shape = ShapeFromFormat(
        params_.data_format, dimensions.batch, dimensions.out_rows,
        dimensions.out_cols, dimensions.out_depth);
    // printf("frontend dimensions batch %d",dimensions.batch);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));
    Tensor im2col;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, TensorShape({dimensions.batch*dimensions.in_depth*dimensions.filter_rows*dimensions.filter_cols*dimensions.out_rows*dimensions.out_cols}), &im2col));
    auto im2col_data = im2col.flat<float>().data();
    auto f_output_data = output->flat<float>().data();
    auto f_input_data = input.flat<float>().data();
    auto f_filter_data = filter.flat<float>().data();

    int filter_left_offset;
    int filter_top_offset;
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
    // Tensor im2ColTmpBuffer;
    // const int64 kMaxChunkSize = (16 * 1024 * 1024) / sizeof(float);

    // {
    //     TensorShape shape;
    //     TensorShapeUtils::MakeShape(&kMaxChunkSize, 1, &shape);
    //     OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, shape, &im2ColTmpBuffer));
    // }
    // printf("leftoffset %d, topoffset %d\n",filter_left_offset,filter_top_offset);
    //printf("filter_Data%f\n",f_input_data[0]);
  double end = realtime();
  #ifdef PROFILE
  cout << "Forward preparation Time difference = " << end - begin  << endl;
  cout << "Forward shape: Input " << dimensions.batch << " " << dimensions.input_rows << " " << dimensions.input_cols << " " << dimensions.in_depth \
  << " Filter " << dimensions.filter_rows << " " << dimensions.filter_cols << " " << dimensions.in_depth << " " << dimensions.out_depth << endl;
  #endif
  
  
    ConvamKernellLauncher(
      context->eigen_device<GPUDevice>(),
      f_input_data,
      f_filter_data,
      im2col_data,
      dimensions.batch,
      dimensions.input_rows,
      dimensions.input_cols,
      dimensions.out_rows,
      dimensions.out_cols,
      dimensions.out_depth,
      dimensions.in_depth,
      dimensions.filter_rows,
      dimensions.filter_cols,
      dimensions.stride_rows,
      dimensions.stride_cols,
      filter_left_offset,
      filter_top_offset,
      params_.padding,
      f_output_data
    );
  }
  private:
    Conv2DParameters params_;
  TF_DISALLOW_COPY_AND_ASSIGN(ConvamOpGPU);
};

REGISTER_KERNEL_BUILDER(Name("Convam").Device(DEVICE_GPU), ConvamOpGPU);



REGISTER_OP("ConvamFilterGrad")
  .Input("input: float")
  .Input("filter_sizes: int32")
  .Input("out_backprop: float")
  .Output("grad_filter: float")
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


class ConvamFilterGradOpCPU: public OpKernel {
public:
  explicit ConvamFilterGradOpCPU(OpKernelConstruction * context): OpKernel(context){
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
    // auto out = filter_backprop->template flat<float>();
    // out.data()[1] = 1;
    // auto grad = out_backprop.flat<float>().data();
    // printf("list grad\n");
    // for(int i = 0; i < filter_shape.num_elements();i++ ){
    //   printf("%f  ",grad[i]);
    // }
    // printf("end of list grad\n");
    // If there is nothing to compute, return.
    if (filter_shape.num_elements() == 0) {
      return;
    }
    // If input is empty, set gradients to zero.
    // if (input.shape().num_elements() == 0) {
    //   functor::SetZeroFunctor<Device, T> f;
    //   f(context->eigen_device<Device>(), filter_backprop->flat<f>());
    //   return;
    // }
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
    // printf("How much top %d\n",(int)pad_top);
    // printf("How much bottom %d\n",(int)pad_bottom);
    // printf("How much left %d\n",(int)pad_left);
    // printf("How much right %d\n",(int)pad_right);
    // For now we take the stride from the second and third dimensions only (we
    // do not support striding on the batch or depth dimension).

    // printf("input_rows/height %d\n",(int)dims.input_size(0));
    // printf("input_cols/width %d\n",(int)dims.input_size(1));
    // printf("input_size_batch %d\n",(int)dims.batch_size);
    // printf("input_size_depth %d\n",(int)dims.in_depth);

    // printf("filter_size0 %d\n",(int)dims.filter_size(0));
    // printf("filter_size1 %d\n",(int)dims.filter_size(1));
    // printf("input_depth %d\n",(int)dims.in_depth);
    // printf("output_depth %d\n",(int)dims.out_depth);


    // printf("outputsize_rows/height %d\n",(int)dims.output_size(0));
    // printf("outputsize_col/width %d\n",(int)dims.output_size(1));
    // printf("");
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
    // printf("How much input_batch %d\n",input_batch);
    // printf("How much input_width %d\n",input_width);
    // printf("How much input_height %d\n",input_height);
    // printf("How much input_channel %d\n",input_channel);
    // printf("How much grad_batch %d\n",grad_batch);
    // printf("How much grad_width %d\n",grad_width);
    // printf("How much grad_height %d\n",grad_height);

    // printf("%d\n",filter_width);
    // printf("%d\n",filter_height);
    // printf("%d\n",filter_indepth);
    // printf("%d\n",filter_outdepth);
    int64 filter_left_offset;
    int64 filter_top_offset;
    //VALID Padding
    if (padding_ == 1) {
      filter_top_offset =
          ((dims.output_size(0) - 1) * stride_rows + dims.filter_size(0) - dims.input_size(0) + 1) /
          2;
      // printf("filter_left_offset = %d, out_rows = %d, stride_cols = %d, filter_rows = %d, input_rows = %d\n",
      //             (int)filter_left_offset,(int)dims.output_size(0),(int)stride_cols,(int)dims.filter_size(0),(int)dims.input_size(0));
      filter_left_offset= ((dims.output_size(1) - 1) * stride_cols + dims.filter_size(1) -
                           dims.input_size(1) + 1) /                  
                          2;
      filter_left_offset = filter_left_offset>0?filter_left_offset:0;
      filter_top_offset = filter_top_offset>0?filter_top_offset:0;
      // printf("filter_top_offset = %d, out_cols = %d, stride_rows = %d, filter_cols = %d, input_cols = %d\n",
      //             (int)filter_top_offset,(int)dims.output_size(1),(int)stride_rows,(int)dims.filter_size(1),(int)dims.input_size(1));
    } else {
      filter_top_offset =
          ((dims.output_size(0) - 1) * stride_rows + dims.filter_size(0) - dims.input_size(0)) / 2;
      // printf("filter_left_offset = %d, out_rows = %d, stride_cols = %d, filter_rows = %d, input_rows = %d\n",
      //             (int)filter_left_offset,(int)dims.output_size(0),(int)stride_cols,(int)dims.filter_size(0),(int)dims.input_size(0));
      filter_left_offset=
          ((dims.output_size(1) - 1) * stride_cols + dims.filter_size(1) - dims.input_size(1)) /
          2;
      // printf("filter_top_offset = %d, out_cols = %d, stride_rows = %d, filter_cols = %d, input_cols = %d\n",
      //             (int)filter_top_offset,(int)dims.output_size(1),(int)stride_rows,(int)dims.filter_size(1),(int)dims.input_size(1));
      filter_left_offset = filter_left_offset>0?filter_left_offset:0;
      filter_top_offset = filter_top_offset>0?filter_top_offset:0;
    }
    // printf("Filter left offset %d, Filter top offset %d\n",(int)filter_left_offset,(int)filter_top_offset);

    // Native backpropogation for filter.
    // printf("How much grad_channel %d\n",grad_channel);
    //grab input data and gradient
    auto grad = out_backprop.flat<float>().data();
    auto in_data = input.flat<float>().data();
    auto out = filter_backprop->template flat<float>().data();

    // now consider our grad become filter output
    for (int out_y = 0; out_y < filter_height; ++out_y) {
      for (int out_x = 0; out_x < filter_width; ++out_x) {
        for (int in_channel = 0; in_channel < filter_indepth; ++in_channel) {
          for (int out_channel = 0; out_channel < filter_outdepth; ++out_channel) {
            const int in_x_origin = (out_x * 1) - filter_left_offset;
            const int in_y_origin = (out_y * 1) - filter_top_offset;
            float total(0);

            // printf("x %d\n",in_x_origin);
            // printf("y %d\n",in_y_origin);
            for (int grad_y = 0; grad_y < (grad_height-1)*stride_rows+1; ++grad_y) {
              for (int grad_x = 0; grad_x < (grad_width-1)*stride_cols+1; ++grad_x) {
                for (int g_batch = 0; g_batch < grad_batch;
                     ++g_batch) {
                  const int in_x = in_x_origin + grad_x;
                  const int in_y = in_y_origin + grad_y;
                  float input_value;
                  // If the location is outside the bounds of the input image,
                  // use zero as a default value.
                  if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                      (in_y < input_height)) {
                    input_value =
                        in_data[(g_batch* input_height * input_width *
                                    input_channel) +
                                   (in_y * input_width * input_channel) +
                                   (in_x * input_channel) + in_channel];
                  } else {
                    input_value = float(0);
                  }
                  
                  const float i_y = (grad_y)/(float)stride_rows ;
                  const float i_x = (grad_x)/(float)stride_cols ;
                  // printf("%f i_y\n",i_y);
                  // printf("%f i_x\n",i_x);
                  // printf("%f fmod i_y\n",fmod(i_y,1));
                  // printf("%f fmod i_x\n",fmod(i_x,1));
                  // printf("%d fmod eq y\n",fmod(i_y,1)==float(0));
                  // printf("%d fmod eq x\n",fmod(i_x,1)==float(0));
                  const bool y = fmod(i_y,1)==float(0);
                  const bool x = fmod(i_x,1)==float(0);

                  const float grad_value = (x&y) ?
                      grad[(g_batch * grad_width * grad_height *
                                   grad_channel) +
                                  (int(i_y) * grad_width*grad_channel) +
                                  (int(i_x) * grad_channel) + out_channel]:0;
                  // printf("grad value %f\n",grad_value);
                  total += (input_value * grad_value);
                }
              }
            }
            out[(out_y * filter_width * filter_indepth *filter_outdepth) +
                        (out_x * filter_indepth *filter_outdepth) +
                        (in_channel *filter_outdepth) + out_channel] = total;

          }
        }
      }
    }

    // std::fstream fs;
    // fs.open("cpu_filter_log", std::fstream::in | std::fstream::out | std::fstream::app);
    // // ----------test for forward propagation-----------
    // for(int i = 0; i < filter_height; i++){
    //   for(int j = 0; j < filter_width; j++){
    //     for(int k = 0; k < filter_indepth; k++){
    //       for(int l = 0; l < filter_outdepth; l++){
    //           // printf("(%d, %d, %d, %d) %f\n",i,j,k,l,output_data[(i * dimensions.out_cols * dimensions.out_rows * dimensions.out_depth) +
    //           //           (j * dimensions.out_cols * dimensions.out_depth) +
    //           //           (k * dimensions.out_depth) + l]);
    //           fs << i <<", " << j << ", " <<k<<", " << l << ",  " << out[(i* filter_width * filter_indepth *filter_outdepth) +
    //                     (j * filter_indepth *filter_outdepth) +
    //                     (k *filter_outdepth) + l]<<"\n";
              
    //       }
    //     }
    //   }
    // }
    // printf("From CPU filter\n");
    // fs << "From CPU\n";
    // fs.close();

  }
  private:
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;
  TF_DISALLOW_COPY_AND_ASSIGN(ConvamFilterGradOpCPU);
 
};
REGISTER_KERNEL_BUILDER(Name("ConvamFilterGrad").Device(DEVICE_CPU), ConvamFilterGradOpCPU);

void ConvamFilterGradKernelLauncher(
      const GPUDevice &d, 
    const float* input,
    const float* grad,
    float* im2col,
    const int input_height,
    const int input_width,
    const int batch,
    const int in_depth,
    const int grad_width,
    const int grad_height,
    const int grad_channel,
    const int filter_left_offset,
    const int filter_top_offset,
    const int stride_row,
    const int stride_col,
    const int filter_width,
    const int filter_height,
    float* out
);
class ConvamFilterGradOpGPU: public OpKernel {
public:
  explicit ConvamFilterGradOpGPU(OpKernelConstruction * context): OpKernel(context){
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
    // TensorShapeUtils::MakeShape(filter_sizes.vec<int32>(), &filter_shape);
    Tensor* filter_backprop = nullptr;
    
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));
                       Tensor* output = nullptr;
    // If there is nothing to compute, return.
    if (filter_shape.num_elements() == 0) {
      return;
    }

    // If input is empty, set gradients to zero.
    // if (input.shape().num_elements() == 0) {
    //   functor::SetZeroFunctor<Device, T> f;
    //   f(context->eigen_device<Device>(), filter_backprop->flat<f>());
    //   return;
    // }
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
      // printf("filter_left_offset = %d, out_rows = %d, stride_cols = %d, filter_rows = %d, input_rows = %d\n",
      //             (int)filter_left_offset,(int)dims.output_size(0),(int)stride_cols,(int)dims.filter_size(0),(int)dims.input_size(0));
      filter_left_offset= ((dims.output_size(1) - 1) * stride_cols + dims.filter_size(1) -
                           dims.input_size(1) + 1) /                  
                          2;
      filter_left_offset = filter_left_offset>0?filter_left_offset:0;
      filter_top_offset = filter_top_offset>0?filter_top_offset:0;
      // printf("filter_top_offset = %d, out_cols = %d, stride_rows = %d, filter_cols = %d, input_cols = %d\n",
      //             (int)filter_top_offset,(int)dims.output_size(1),(int)stride_rows,(int)dims.filter_size(1),(int)dims.input_size(1));
    } else {
      filter_top_offset =
          ((dims.output_size(0) - 1) * stride_rows + dims.filter_size(0) - dims.input_size(0)) / 2;
      // printf("filter_left_offset = %d, out_rows = %d, stride_cols = %d, filter_rows = %d, input_rows = %d\n",
      //             (int)filter_left_offset,(int)dims.output_size(0),(int)stride_cols,(int)dims.filter_size(0),(int)dims.input_size(0));
      filter_left_offset=
          ((dims.output_size(1) - 1) * stride_cols + dims.filter_size(1) - dims.input_size(1)) /
          2;
      // printf("filter_top_offset = %d, out_cols = %d, stride_rows = %d, filter_cols = %d, input_cols = %d\n",
      //             (int)filter_top_offset,(int)dims.output_size(1),(int)stride_rows,(int)dims.filter_size(1),(int)dims.input_size(1));
      filter_left_offset = filter_left_offset>0?filter_left_offset:0;
      filter_top_offset = filter_top_offset>0?filter_top_offset:0;

    }
    // printf("Filter left offset %d, Filter top offset %d\n",(int)filter_left_offset,(int)filter_top_offset);
    Tensor im2col;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, 
    TensorShape({input_batch*input_channel*((grad_height-1)*stride_rows+1)*((grad_width-1)*stride_cols+1)*filter_width*filter_height})
    , &im2col));
    auto im2col_data = im2col.flat<float>().data();
    auto grad = out_backprop.flat<float>().data();
    auto in_data = input.flat<float>().data();
    auto out = filter_backprop->template flat<float>().data();
    double end = realtime();
  #ifdef PROFILE
   cout << "Backprop filter grad Time difference = " << end - begin << endl;
   cout << "Backpropfilter shape: Input " << input_batch << " " << input_height << " " << input_width << " " << input_channel \
  << " Output " << input_batch << " " << grad_height << " " << grad_width << " " << grad_channel << endl;
   #endif
    ConvamFilterGradKernelLauncher(
            context->eigen_device<GPUDevice>(),
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
      filter_top_offset,
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
REGISTER_KERNEL_BUILDER(Name("ConvamFilterGrad").Device(DEVICE_GPU).HostMemory("filter_sizes"), ConvamFilterGradOpGPU);



REGISTER_OP("ConvamInputGrad")
    .Input("input_sizes: int32")
    .Input("filter: float")
    .Input("out_backprop: float")
    .Output("grad_input: float")
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
class ConvamInputGradOpCPU : public OpKernel {
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
    
    // If there is nothing to compute, return.
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
    // printf("input size %d\n",input_sizes.vec<int32>().data()[0]);
    // printf("input size %d\n",input_sizes.vec<int32>().data()[1]);
    // printf("input size %d\n",input_sizes.vec<int32>().data()[2]);
    // printf("input size %d\n",input_sizes.vec<int32>().data()[3]);

    // For now we take the stride from the second and third dimensions only (we
    // do not support striding on the batch or depth dimension).
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
    const int back_pad_bottom = input_height - (grad_height-1)*stride_rows-2-back_pad_top+filter_height;
    const int back_pad_right = input_width - (grad_width-1)*stride_cols-2-back_pad_left+filter_width;

    // printf("back_pad_top %d\n",back_pad_top);
    // printf("back_pad_left %d\n",back_pad_left);
    // printf("back_pad_bottom %d,input_height %d, grad_height %d, stride_rows %d,filter_height %d \n",
    // back_pad_bottom,input_height,grad_height,stride_rows,filter_height);
    // printf("back_pad_right %d\n",back_pad_right);

        //grab input and output
    auto grad = out_backprop.flat<float>().data();
    auto in_data = filter.flat<float>().data();
    auto out = in_backprop->template flat<float>().data();


    for (int batch = 0; batch < input_batch; ++batch) {
      for (int out_y = 0; out_y < input_height; ++out_y) {
        for (int out_x = 0; out_x < input_width; ++out_x) {
          for (int in_channel = 0; in_channel < input_channel; ++in_channel) {
            const int in_x_origin = out_x  - back_pad_left;
            const int in_y_origin = out_y  - back_pad_top;  
            // printf("in_x %d\n",in_x_origin);
            // printf("in_y %d\n",in_y_origin);
            float total(0);
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                for (int out_channel = 0; out_channel < output_channel;
                     ++out_channel) {
                    const int in_x = in_x_origin + filter_x;
                    const int in_y = in_y_origin + filter_y;
                    // printf("in_x %d\n",in_x);
                    // printf("in_y %d\n",in_y);
                    float input_value;
                    if((in_x >= 0) && (in_x < hole_grad_width) && (in_y >= 0) &&
                      (in_y < hole_grad_height)){
                        const float i_y = (in_y)/(float)stride_rows ;
                        const float i_x = (in_x)/(float)stride_cols ;
                        const bool y = fmod(i_y,1)==float(0);
                        const bool x = fmod(i_x,1)==float(0);
                        input_value = (x&y) ?
                      grad[(batch *grad_width*grad_height*output_channel) +
                                  (int(i_y) * grad_width*output_channel) +
                                  (int(i_x) * output_channel) + out_channel]:0;

                    } else {
                      input_value = float(0);
                    }

                    const float filter = in_data[((filter_height-1-filter_y) * filter_width * input_channel*output_channel) +
                                  ((filter_width-1-filter_x) * input_channel*output_channel) +
                                  (in_channel * output_channel) + out_channel];

                    total += (input_value*filter);
                
                }
              }
            }
            out[(batch * input_width * input_height * input_channel) +
                        (out_y * input_width * input_channel) +
                        (out_x *input_channel) + in_channel] = total;
          }
        }
      }
    }

  // std::fstream fs;
  //   fs.open("cpu_input_log", std::fstream::in | std::fstream::out | std::fstream::app);
  //   // ----------test for forward propagation-----------
  //   for(int i = 0; i < input_batch; i++){
  //     for(int j = 0; j < input_height; j++){
  //       for(int k = 0; k < input_width; k++){
  //         for(int l = 0; l < input_channel; l++){
  //             // printf("(%d, %d, %d, %d) %f\n",i,j,k,l,output_data[(i * dimensions.out_cols * dimensions.out_rows * dimensions.out_depth) +
  //             //           (j * dimensions.out_cols * dimensions.out_depth) +
  //             //           (k * dimensions.out_depth) + l]);
  //             fs << i <<", " << j << ", " <<k<<", " << l << ",  " << out[(i* input_height * input_width *input_channel) +
  //                       (j * input_width *input_channel) +
  //                       (k *input_channel) + l]<<"\n";
              
  //         }
  //       }
  //     }
  //   }
  //   printf("From CPU Input\n");
  //   fs << "From CPU\n";
  //   fs.close();
  }

  private:
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;

  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(ConvamInputGradOpCPU);
};




REGISTER_KERNEL_BUILDER(Name("ConvamInputGrad").Device(DEVICE_CPU), ConvamInputGradOpCPU);


void ConvamInputGradKernelLauncher(
      // grad needs pading and holes
      // im2col input
      const GPUDevice &d,
      const float* grad,
      float* holed_grad,
      float* im2col,
      const int real_grad_height,
      const int real_grad_width,
      const int hole_grad_width,
      const int hole_grad_height,
      const int back_pad_top,
      const int back_pad_left,
      const int back_pad_bottom,
      const int back_pad_right,
      // reverse
      const float* filter,
      float* rsfilter,
      const int filter_height,
      const int filter_width,
      const int output_channel,
      const int stride_rows,
      const int stride_cols,
      // input related
      const int input_batch,
      const int input_height,
      const int input_width,
      const int input_channel,
      float* output
    );


class ConvamInputGradOpGPU : public OpKernel {
 public:
  explicit ConvamInputGradOpGPU(OpKernelConstruction* context)
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
    double begin = realtime();
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
    
    // If there is nothing to compute, return.
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
    const int back_pad_bottom = input_height - (grad_height-1)*stride_rows-2-back_pad_top+filter_height;
    const int back_pad_right = input_width - (grad_width-1)*stride_cols-2-back_pad_left+filter_width;
    // std::cout << "back_pad_top " << back_pad_top << "back_pad_left " <<back_pad_left << "back_pad_bottom " << back_pad_bottom<< "back_pad_right " << back_pad_right<< "\n";
    Tensor holed_grad;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, 
    TensorShape({hole_grad_height*hole_grad_width*output_channel*grad_batch})
    , &holed_grad));

    Tensor im2col;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, 
    TensorShape({output_channel*filter_width*filter_height*input_height*input_width*grad_batch})
    , &im2col));
    Tensor rsfilter;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, 
    TensorShape({output_channel*filter_width*filter_height*input_channel})
    , &rsfilter));
    auto grad = out_backprop.flat<float>().data();
    auto in_data = filter.flat<float>().data();
    auto out = in_backprop->template flat<float>().data();
    auto holed_grad_data = holed_grad.flat<float>().data();
    auto im2_col_data = im2col.flat<float>().data();
    auto rsfilter_data = rsfilter.flat<float>().data();
    double end = realtime();

    #ifdef PROFILE
     cout << "backprop input grad preparation Time difference = " << end - begin << endl;
     cout << "Backpropinput shape: Output " << input_batch << " " << hole_grad_height << " " << hole_grad_width << " " << output_channel \
  << " filter " << filter_height<< " " << filter_width << " " << input_channel << " " << output_channel << endl;
     #endif
    ConvamInputGradKernelLauncher(
        // grad needs pading and holes
        // im2col input
        ctx->eigen_device<GPUDevice>(),
        grad,
        holed_grad_data,
        im2_col_data,
        grad_height,

        grad_width,
        hole_grad_width,
        hole_grad_height,
        back_pad_top,
        back_pad_left,
        back_pad_bottom,
        back_pad_right,
        // reverse
        in_data,
        rsfilter_data,
        filter_height,
        filter_width,
        output_channel,
        stride_rows,
        stride_cols,
        // input related
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
REGISTER_KERNEL_BUILDER(Name("ConvamInputGrad").Device(DEVICE_GPU).HostMemory("input_sizes"), ConvamInputGradOpGPU);
