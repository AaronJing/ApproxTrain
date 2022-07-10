#include "convam.h"
#include "approx_mul_lut.h"
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
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/types.h"
#include <math.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <sys/time.h>

using namespace std;
using namespace tensorflow;
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;


/*************************** from original tensorflow source ******************************/

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



/****************************  start of lut cpu specialization **********************************/
template<>
class approx_mul_lut<CPUDevice> : public approx_mul_lut_base {
    public:
        explicit approx_mul_lut(OpKernelConstruction *context) : approx_mul_lut_base(
                context
                ) {};
};
/****************************  end of lut cpu specialization **********************************/


/**************************** Our custom op implementation **********************************/

/*---------------------------- custom convolution operation --------------------*/

REGISTER_OP("Convam")
    .Input("input: T")
    .Input("filter: T")
    .Output("output: T")
    .Attr("T: {float, int32}")
    .Attr("strides: list(int)")
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("mant_mul_lut: string")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .SetShapeFn(::tensorflow::shape_inference::Conv2DShape);

// cpu specilisation
template <typename T>
struct ConvamFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, const T* input_data, T* output_data,
            const int batch, const int out_rows, const int out_cols,
            const int out_depth, const int stride_cols, const int stride_rows,
            const int filter_left_offset, const int filter_top_offset,
            const int filter_rows, const int filter_cols, const int in_depth,
            const int input_cols, const int input_rows, const T* filter,
            const T* im2col, const int padding,
            approx_mul_lut<CPUDevice>& mul_lut
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
                      filter[(filter_y * filter_cols * in_depth *
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
class ConvamOp : public OpKernel{
public:
  explicit ConvamOp(OpKernelConstruction* context) : OpKernel(context), 
    mul_lut_(context) {
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
    //int64 const oneinputsize = dimensions.input_rows * dimensions.input_cols * dimensions.in_depth;
    //int64 const oneoutputsize = dimensions.out_rows* dimensions.out_cols * dimensions.out_depth;
    if(dimensions.filter_cols == 1 && dimensions.filter_rows == 1 &&
            dimensions.stride_rows == 1 && dimensions.stride_cols){
        size_t const block_size = 16;
        size_t const max_batch = (65536*block_size + 1 - block_size)/(dimensions.input_rows*dimensions.input_cols);
        if ((size_t)dimensions.batch <= max_batch) {
            long long int size_shape = dimensions.batch*dimensions.in_depth*dimensions.filter_rows*dimensions.filter_cols*dimensions.out_rows*dimensions.out_cols;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(), TensorShape({size_shape}), &im2col));
        } else {
            long long int size_shape = max_batch*dimensions.in_depth*dimensions.filter_rows*dimensions.filter_cols*dimensions.out_rows*dimensions.out_cols ;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(), TensorShape({size_shape}), &im2col));
        }
    } else if(dimensions.filter_rows == dimensions.input_rows && dimensions.filter_cols == dimensions.input_cols&& params_.padding == 1){
        long long int size_shape = dimensions.batch*dimensions.in_depth*dimensions.filter_rows*dimensions.filter_cols*dimensions.out_rows*dimensions.out_cols;
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(), TensorShape({size_shape}), &im2col));
    } else {
        size_t const block_size = 16;
        size_t const max_batch = (65536*block_size + 1 - block_size)/(dimensions.out_rows*dimensions.out_cols);
        if ((size_t)dimensions.batch <= max_batch) {
            long long int size_shape_x = dimensions.batch*dimensions.out_cols*dimensions.out_rows;
            long long int size_shape_y = dimensions.in_depth*dimensions.filter_rows*dimensions.filter_cols;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(), TensorShape({size_shape_x,size_shape_y}), &im2col));
        } else {
            long long int size_shape = max_batch*dimensions.in_depth*dimensions.filter_rows*dimensions.filter_cols*dimensions.out_rows*dimensions.out_cols;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(), TensorShape({size_shape}), &im2col));
        }
    }
    // allocate gpu mem for lut
    
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
            dimensions.out_cols,
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
            im2col_data,
            params_.padding,
            mul_lut_
            );
  }
  private:
  approx_mul_lut<Device> mul_lut_;
  Conv2DParameters params_;
  TF_DISALLOW_COPY_AND_ASSIGN(ConvamOp);
};

// For forwardpropagation
// Register the CPU kernels.
#define REGISTER_CPU_CONVAM(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Convam").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ConvamOp<CPUDevice, T>);
REGISTER_CPU_CONVAM(float);
REGISTER_CPU_CONVAM(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU_CONVAM(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template class ConvamFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Convam").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ConvamOp<GPUDevice, T>);
REGISTER_GPU_CONVAM(float);
REGISTER_GPU_CONVAM(int32);
#endif  // GOOGLE_CUDA



/*---------------------------- custom convolution filter gradient --------------------*/

// CPU specisialization for Convam filter grad
template <typename T>
struct ConvamFilterGradFunctor<CPUDevice, T>{
  void operator()(const CPUDevice& d, const T* input, const T* grad, T* im2col,
          const int input_rows, const int input_cols, const int batch,
          const int in_depth, const int out_cols, const int out_rows,
          const int out_depth, const int filter_left_offset,
          const int filter_top_offset, const int stride_rows,
          const int stride_cols, const int filter_cols, const int filter_rows,
          T* output, approx_mul_lut<CPUDevice>& mul_lut
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
                                    input_value = input[
                                        (g_batch* input_rows * input_cols *
                                         in_depth) +
                                        (in_y * input_cols * in_depth) +
                                        (in_x * in_depth) + in_channel];
                                } else {
                                    input_value = T(0);
                                }

                                const float i_y = (grad_y)/(float)stride_rows ;
                                const float i_x = (grad_x)/(float)stride_cols ;
                                const bool y = fmod(i_y,1)==float(0);
                                const bool x = fmod(i_x,1)==float(0);

                                const T grad_value = (x&y) ?
                                    grad[(g_batch * out_cols * out_rows *
                                    out_depth) +
                                    (int(i_y) * out_cols*out_depth) +
                                    (int(i_x) * out_depth) + out_channel]:0;
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

REGISTER_OP("ConvamFilterGrad")
  .Input("filter_sizes: int32")
  .Input("input: T")
  .Input("out_backprop: T")
  .Output("grad_filter: T")
  .Attr("T: {float, int32}")
  .Attr("strides: list(int)")
  .Attr("dilations: list(int) = [1, 1, 1, 1]")
  .Attr("mant_mul_lut: string")
  .Attr(GetPaddingAttrString())
  .Attr(GetConvnetDataFormatAttrString())
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    });

template <typename Device, typename T>
class ConvamFilterGradOp: public OpKernel {
public:
  explicit ConvamFilterGradOp(OpKernelConstruction * context): OpKernel(context), mul_lut_(context){
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
    const Tensor& input = context->input(1);
    const Tensor& filter_sizes = context->input(0);
    const Tensor& out_backprop = context->input(2);

    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(filter_sizes.shape()),
        errors::InvalidArgument(
            "Conv2DBackpropFilter: filter_sizes input must be 1-dim, not ",
            filter_sizes.dims()));

    auto filterdimdata = filter_sizes.vec<int32>().data();

    TensorShape filter_shape = {filterdimdata[0], filterdimdata[1], filterdimdata[2],filterdimdata[3]};
    Tensor* filter_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &filter_backprop));
                       //Tensor* output = nullptr;
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
    //const int dilation_rows = GetTensorDim(dilations_, data_format_, 'H');
    //const int dilation_cols = GetTensorDim(dilations_, data_format_, 'W');
    //get input dim
    const int input_batch = GetTensorDim(input,data_format_,'N');
    const int input_width = GetTensorDim(input,data_format_,'W');
    const int input_height = GetTensorDim(input,data_format_,'H');
    const int input_channel = GetTensorDim(input,data_format_,'C');
    //get grad dim
    //const int grad_batch = GetTensorDim(out_backprop,data_format_,'N');
    const int grad_width = GetTensorDim(out_backprop,data_format_,'W');
    const int grad_height = GetTensorDim(out_backprop,data_format_,'H');
    const int grad_channel = GetTensorDim(out_backprop,data_format_,'C');
    //get filter dim
    const int filter_width = filter_shape.dim_size(1);
    const int filter_height = filter_shape.dim_size(0);
    //const int filter_indepth = filter_shape.dim_size(2);
    //const int filter_outdepth = filter_shape.dim_size(3);

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
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
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
            filter_top_offset,
            stride_rows,
            stride_cols,
            filter_width,
            filter_height,
            out,
            mul_lut_
            );
  }
  private:
  approx_mul_lut<Device> mul_lut_;
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;
  TensorFormat data_format_;
  TF_DISALLOW_COPY_AND_ASSIGN(ConvamFilterGradOp);
};

// For backpropagation:filter
// Register the CPU kernels.
#define REGISTER_CPU_FILTERGRAD(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ConvamFilterGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ConvamFilterGradOp<CPUDevice, T>);
REGISTER_CPU_FILTERGRAD(float);
REGISTER_CPU_FILTERGRAD(int32);
// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU_FILTERGRAD(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template class ConvamFilterGradFunctor<GPUDevice, T>; \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ConvamFilterGrad").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("filter_sizes"), \
      ConvamFilterGradOp<GPUDevice, T>);
REGISTER_GPU_FILTERGRAD(float);
REGISTER_GPU_FILTERGRAD(int32);
#endif  // GOOGLE_CUDA

/*---------------------------- custom convolution input gradient --------------------*/

// CPU specisialization for ConvamInputGrad
template <typename T>
struct ConvamInputGradFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, const T* grad, T* im2col,
          const int hole_grad_width, const int hole_grad_height,
          const int pad_top, const int pad_left, const T* filter, T* rsfilter,
          const int filter_rows, const int filter_cols, const int out_depth,
          const int stride_rows, const int stride_cols, const int batch,
          const int input_rows, const int input_cols, const int in_depth,
          T* output, const int out_rows, const int out_cols,
          approx_mul_lut<CPUDevice>& mul_lut
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
                                    grad[(batch *out_cols*out_rows*out_depth) +
                                  (int(i_y) * out_cols*out_depth) +
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
    .Input("filter: T")
    .Input("out_backprop: T")
    .Output("grad_input: T")
    .Attr("T: {float, int32}")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .Attr("mant_mul_lut: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      ::tensorflow::shape_inference::ShapeHandle s;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
      TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
      c->set_output(0, s);
      return Status::OK();
    });
template <typename Device, typename T>
class ConvamInputGradOp: public OpKernel {
    public:
     explicit ConvamInputGradOp(OpKernelConstruction* context): OpKernel(context),mul_lut_(context) {
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
        auto indimdata = input_sizes.vec<int32>().data();
        TensorShape input_shape = {indimdata[0], indimdata[1], indimdata[2], indimdata[3]};
        //OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
          //                          input_sizes.vec<int32>(), &input_shape));

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
        //const int grad_channel = GetTensorDim(out_backprop,data_format_,'C');

        const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
        const int stride_cols = GetTensorDim(strides_, data_format_, 'W');
        //const int dilation_rows = GetTensorDim(dilations_, data_format_, 'H');
        //const int dilation_cols = GetTensorDim(dilations_, data_format_, 'W');

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
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
        TensorShape({output_channel*filter_width*filter_height*input_channel})
        , &rsfilter));
        auto grad = out_backprop.flat<T>().data();
        auto in_data = filter.flat<T>().data();
        auto out = in_backprop->template flat<T>().data();
        auto rsfilter_data = rsfilter.flat<T>().data();

       // auto const oneinputsize = input_height*input_width*input_channel;
        //auto const oneoutputsize = grad_height*grad_width*output_channel;
        size_t const block_size = 16;
        size_t const max_batch = (65536*block_size + 1 - block_size)/(input_height*input_width);
        Tensor im2col;
        if ((size_t)input_batch <= max_batch) {
            long long int size_shape = output_channel*filter_width*filter_height*input_height*input_width*grad_batch;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
            TensorShape({size_shape})
            , &im2col));
        } else {
            long long int size_shape = output_channel*filter_width*filter_height*input_height*input_width*max_batch;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
            TensorShape({size_shape})
            , &im2col));
        }
        auto im2col_data = im2col.flat<T>().data();
        ConvamInputGradFunctor<Device, T>()(
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
                out,
                grad_height,
                grad_width,
                mul_lut_
                );
     }
  private:
  approx_mul_lut<Device> mul_lut_;
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  Padding padding_;

  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(ConvamInputGradOp);
};

// For backpropagation:input
// Register the CPU kernels.
#define REGISTER_CPU_INPUTGRAD(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ConvamInputGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ConvamInputGradOp<CPUDevice, T>);
REGISTER_CPU_INPUTGRAD(float);
REGISTER_CPU_INPUTGRAD(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU_INPUTGRAD(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template class ConvamInputGradFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ConvamInputGrad").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("input_sizes"), \
      ConvamInputGradOp<GPUDevice, T>);
REGISTER_GPU_INPUTGRAD(float);
REGISTER_GPU_INPUTGRAD(int32);
#endif  // GOOGLE_CUDA
