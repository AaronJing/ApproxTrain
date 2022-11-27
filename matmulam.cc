
/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/matmul_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/util/matmul_autotune.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
//#if GOOGLE_CUDA
//#include "third_party/gpus/cuda/include/cuda.h"
//#endif
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/gpu_utils.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;



REGISTER_OP("MatMulAM")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr(
        "T: {bfloat16, half, float, double, int32, int64, complex64, "
        "complex128}")
    .SetShapeFn(shape_inference::MatMulShape);



template <typename T>
struct LaunchMatMul<GPUDevice, T> {
  static void launch(
      OpKernelContext* ctx, const Tensor& a, const Tensor& b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      std::vector<int64>* algorithms, bool use_autotune, Tensor* out) {
};

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Device, typename T, bool USE_CUBLAS>
class MatMulAMOp : public OpKernel {
 public:
  explicit MatMulAMOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), algorithms_set_already_(false) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));

    LaunchMatMul<Device, T, USE_CUBLAS>::GetBlasGemmAlgorithm(
        ctx, &algorithms_, &algorithms_set_already_);
    use_autotune_ = MatmulAutotuneEnable();
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);
    if (a.dims() == 2) {
        OP_REQUIRES(ctx, b.dims() == 2, errors::InvalidArgument("Input b shold be 2 dimensional"));
        OP_REQUIRES(ctx, a.shape().dim_size(1) == b.shape().dim_size(0), errors::InvalidArgument("Does not meet cols of mat a is equal rows of mat b"));
    } else if (a.dims() == 3) {
        OP_REQUIRES(ctx, b.dims() == 3, errors::InvalidArgument("Input b shold be 3 dimensional"));
        OP_REQUIRES(ctx, a.shape().dim_size(0) == b.shape().dim_size(0), errors::InvalidArgument("Does not meet batch of mat a is equal batch of matb"));
        OP_REQUIRES(ctx, a.shape().dim_size(2) == b.shape().dim_size(1), errors::InvalidArgument("Does not meet cols of mat a is equal rows of mat b"));
        OP_REQUIRES(ctx, a.shape().dim_size(2) == b.shape().dim_size(1), errors::InvalidArgument("Does not meet cols of mat a is equal rows of mat b"));
    } else {
        OP_REQUIRES(
                ctx, false,
                errors::InvalidArgument("Input a and b should be 2 dimensional or 3 dimensional")
                );
    }

    TensorShape out_shape(
        {a.dim_size(0), a.dim_size(0), b.dim_size(1)});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }

    if (a.NumElements() == 0 && b.NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the
      // output shape is [x, y] where x and y are non-zero, so we fill
      // the output with zeros.
      std::memset(out->flat<T>(), 0, sizeof(T)*out->NumElements())
      return;
    }

    LaunchMatMul<Device, T, USE_CUBLAS>::launch(
          ctx, a, b, dim_pair, &algorithms_, use_autotune_, out);

 private:
  std::vector<int64> algorithms_;
  bool algorithms_set_already_;
  bool use_autotune_;
  bool transpose_a_;
  bool transpose_b_;
};

#define REGISTER_GPU(T)                                            \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("MatMulAM").Device(DEVICE_GPU).TypeConstraint<T>("T"),    \
      MatMulAMOp<GPUDevice, T, true /* cublas, true by default */>); \

REGISTER_GPU(float);
}  // namespace tensorflow
