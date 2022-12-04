
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

#include "approx_mul_lut.h"
#include "matmulam.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;



REGISTER_OP("MatMulAM")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {float, int32}")
    .Attr("mant_mul_lut: string")
    .SetShapeFn(shape_inference::MatMulShape);


template<>
class approx_mul_lut<CPUDevice> : public approx_mul_lut_base {
    public:
        explicit approx_mul_lut(OpKernelConstruction *context) : approx_mul_lut_base(
                context
                ) {};
};



template <typename Device, typename T>
class MatMulAMOp : public OpKernel {
 public:
  explicit MatMulAMOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), mul_lut_(ctx){

  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);
    int batch_a = 0;
    // quick fix for experiment, we should support batch after revision
    int batch_a_outer = 0;
    int batch_b = 0;
    // quick fix for experiment, we should support batch after revision
    int batch_b_outer =0;
    int row_a = 0;
    int col_a = 0;
    int row_b = 0;
    int col_b = 0;
    if (a.dims() == 2 && b.dims() == 2) {
        OP_REQUIRES(ctx, a.shape().dim_size(1) == b.shape().dim_size(0), errors::InvalidArgument("Does not meet cols of mat a is equal rows of mat b"));
        row_a = a.shape().dim_size(0);
        col_a = a.shape().dim_size(1);
        row_b = b.shape().dim_size(0);
        col_b = b.shape().dim_size(1);
    } else if (a.dims() == 3 && b.dims() == 3) {
        OP_REQUIRES(ctx, a.shape().dim_size(0) == b.shape().dim_size(0) or a.shape().dim_size(0) == 1 or b.shape().dim_size(0) == 1, errors::InvalidArgument("Does not meet batch of mat a is equal batch of matb"));
        OP_REQUIRES(ctx, a.shape().dim_size(2) == b.shape().dim_size(1), errors::InvalidArgument("Does not meet cols of mat a is equal rows of mat b"));
        batch_a = a.shape().dim_size(0);
        batch_b = b.shape().dim_size(0);
        row_a = a.shape().dim_size(1);
        col_a = a.shape().dim_size(2);
        row_b = b.shape().dim_size(1);
        col_b = b.shape().dim_size(2);
    } else if (b.dims() == 3 && a.dims() == 2) {
        OP_REQUIRES(ctx, a.shape().dim_size(1) == b.shape().dim_size(1), errors::InvalidArgument("Does not meet cols of mat a is equal rows of mat b"));
        batch_b = b.shape().dim_size(0);    
        row_a = a.shape().dim_size(0);
        col_a = a.shape().dim_size(1);
        row_b = b.shape().dim_size(1);
        col_b = b.shape().dim_size(2);
    } else if (b.dims() == 2 && a.dims() == 3) {
        OP_REQUIRES(ctx, a.shape().dim_size(2) == b.shape().dim_size(1), errors::InvalidArgument("Does not meet cols of mat a is equal rows of mat b"));
        batch_a = a.shape().dim_size(0);    
        row_a = a.shape().dim_size(1);
        col_a = a.shape().dim_size(2);
        row_b = b.shape().dim_size(0);
        col_b = b.shape().dim_size(1);
    } else if (b.dims()==4 && a.dims()==4){
        OP_REQUIRES(ctx, a.shape().dim_size(0) == b.shape().dim_size(0), errors::InvalidArgument("Does not meet outer batch of mat a is equal outer batch of matb"));
        OP_REQUIRES(ctx, a.shape().dim_size(1) == b.shape().dim_size(1), errors::InvalidArgument("Does not meet batch of mat a is equal batch of matb"));
        OP_REQUIRES(ctx, a.shape().dim_size(3) == b.shape().dim_size(2), errors::InvalidArgument("Does not meet cols of mat a is equal rows of mat b"));
        batch_a_outer = a.shape().dim_size(0);    
        batch_a = a.shape().dim_size(1);    
        batch_b_outer = b.shape().dim_size(0);    
        batch_b = b.shape().dim_size(1);    

        row_a = a.shape().dim_size(2);
        col_a = a.shape().dim_size(3);
        row_b = b.shape().dim_size(2);
        col_b = b.shape().dim_size(3);
    
    } else {
        OP_REQUIRES(
                ctx, false,
                errors::InvalidArgument("Input a and b should be 2 dimensional or 3 dimensional and compitable batch size")
                );
    }

    Tensor* out = nullptr;
    if (batch_a_outer == 0 && batch_b_outer == 0){
        if (batch_a!=0 || batch_b!=0) {
            if (batch_a != 0 && batch_b != 0) {
                if (batch_a == 1 || batch_b == 1){
                    if (batch_a > batch_b) {
                        TensorShape out_shape({a.dim_size(0), a.dim_size(1), b.dim_size(2)});
                        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
                    } else {
                        TensorShape out_shape({b.dim_size(0), a.dim_size(1), b.dim_size(2)});
                        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
                    } 
                } else {
                    TensorShape out_shape({a.dim_size(0), a.dim_size(1), b.dim_size(2)});
                    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
                }
            } else if ( batch_a != 0) {
                TensorShape out_shape({a.dim_size(0), a.dim_size(1), b.dim_size(1)});
                OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
            } else {
                TensorShape out_shape({b.dim_size(0), a.dim_size(0), b.dim_size(2)});
                OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
            }
        } else { 
            TensorShape out_shape1({a.dim_size(0), b.dim_size(1)});
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape1, &out));
        }
    } else {
        TensorShape out_shape({b.dim_size(0), b.dim_size(1), a.dim_size(2), b.dim_size(3)});
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
        batch_a = batch_a * batch_a_outer;
        batch_b = batch_b * batch_b_outer;
    }
    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;

    if (a.NumElements() == 0 && b.NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the
      // output shape is [x, y] where x and y are non-zero, so we fill
      // the output with zeros.
      std::memset(out->flat<T>().data(), 0, sizeof(T)*out->NumElements());
      return;
    }

  //static void launch(
  //    OpKernelContext* ctx, const T& a, const T& b,
  //    const int batch, const int row_a, const int col_a, const int row_b,
  //    const int col_b, T* out) {
    auto a_data = a.flat<T>().data();
    auto b_data = b.flat<T>().data();
    auto output = out->flat<T>().data();
    LaunchMatMul<Device, T>()(
          ctx->eigen_device<Device>(), a_data, b_data, batch_a, batch_b, row_a, col_a, row_b, col_b, output, mul_lut_ );

 }
 private:
    approx_mul_lut<Device> mul_lut_;
    TF_DISALLOW_COPY_AND_ASSIGN(MatMulAMOp);
};
template<typename T>
void gemm_cpu(
        const CPUDevice &d, int m, int n, int k, const T *a, int lda,
        const T *b, int ldb,
        T *c, int ldc
        )
{
    const size_t aIStride = size_t(lda);
    const size_t aLStride = 1;
    const size_t bJStride = 1;
    const size_t bLStride = size_t(ldb);
    const size_t cIStride = size_t(ldc);
    const size_t cJStride = 1;

    for(size_t j = 0; j < size_t(n); ++j)
    {
        for(size_t i = 0; i < size_t(m); ++i)
        {
            T total(0);
            for(size_t l = 0; l < size_t(k); ++l)
            {
                const size_t aIndex = ((i * aIStride) + (l * aLStride));
                const T aValue = a[aIndex];
                const size_t bIndex = ((j * bJStride) + (l * bLStride));
                const T bValue = b[bIndex];
                total += (aValue * bValue);
            }
            const size_t cIndex = ((i * cIStride) + (j * cJStride));
            c[cIndex] = total;
        }
    }
};
template <typename T>
struct LaunchMatMul<CPUDevice, T> {
  void operator()(
      const CPUDevice &d, const T& a, const T& b,
      const int batch_a, const int batch_b, const int row_a, const int col_a, const int row_b,
      const int col_b, T* out,
      approx_mul_lut<CPUDevice>& mul_lut
      ) {
      int m = row_a;
      int n = col_b;
      int k = col_a;
      int lda = col_a;
      int ldb = col_b;
      int ldc = col_b;
      if (batch_a != 0 && batch_b != 0) {
        if (batch_a > batch_b) {
            for (int i = 0; i < batch_a; i++){
                // strided gemm?
                const T& temp_a = a + i*row_a*col_a;
                T& temp_c = out + i*row_a*col_b;
                gemm_cpu<T>(d, m, n, k, temp_a, lda, b, ldb, temp_c, ldc);
            }
        } else if (batch_a != 1 && batch_b != 1) {
            for (int i = 0; i < batch_b; i++){
                // strided gemm?
                const T& temp_a = a + i*row_a*col_a;
                const T& temp_b = b + i*row_b*col_b;
                T& temp_c = out + i*row_a*col_b;
                gemm_cpu<T>(d, m, n, k, temp_a, lda, temp_b, ldb, temp_c, ldc);
            }
        
        } else {
            for (int i = 0; i < batch_b; i++){
                // strided gemm?
                const T& temp_b = b + i*row_b*col_b;
                T& temp_c = out + i*row_a*col_b;
                gemm_cpu<T>(d, m, n, k, a, lda, temp_b, ldb, temp_c, ldc);
            }
        
        }
      } else if (batch_a!=0){
        for (int i = 0; i < batch_a; i++){
            // strided gemm?
            const T& temp_a = a + i*row_a*col_a;
            T& temp_c = out + i*row_a*col_b;
            gemm_cpu<T>(d, m, n, k, temp_a, lda, b, ldb, temp_c, ldc);
        }
      } else if (batch_b!=0) {
        for (int i = 0; i < batch_b; i++){
            // strided gemm?
            const T& temp_b = a + i*row_b*col_b;
            T& temp_c = out + i*row_a*col_b;
            gemm_cpu<T>(d, m, n, k, a, lda, temp_b, ldb, temp_c, ldc);
        }
      } else {
        gemm_cpu<T>(d, m, n, k, a, lda, b, ldb, out, ldc);
      }
  }
};

#define REGISTER_GPU(T)                                            \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("MatMulAM").Device(DEVICE_GPU).TypeConstraint<T>("T"),    \
      MatMulAMOp<GPUDevice, T>); \

REGISTER_GPU(float)
