
#include "denseam.h"
#include "approx_mul_lut.h"
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
template<>
class approx_mul_lut<CPUDevice> : public approx_mul_lut_base {
    public:
        explicit approx_mul_lut(OpKernelConstruction *context) : approx_mul_lut_base(
                context
                ) {};
};
REGISTER_OP("Denseam")
  .Input("input: T")
  .Input("weights: T")
  .Output("output: T")
  .Attr("T: {float, int32}")
  .Attr("mant_mul_lut: string")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

   shape_inference::ShapeHandle input_shape;
   TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));

   shape_inference::ShapeHandle weight_shape;
   TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &weight_shape));
   
   
  shape_inference::DimensionHandle samples = c->Dim(input_shape, 0);
  shape_inference::DimensionHandle units = c->Dim(weight_shape, 1);
  
  c->set_output(0, c->Matrix(samples, units));

    return Status::OK();
  });
template <typename T>
struct DenseamFunctor<CPUDevice, T>{
    void operator()(const CPUDevice& d, const T* input, const T* weights,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<CPUDevice>& mul_lut
            ){
            for(int i = 0; i < batch; i++){
                for(int j = 0; j < units; j++){
                    output[i*units + j] = T(0);
                    for(int k = 0; k < input_width; k++){
                        output[i*units + j] += input[i*input_width + k] * weights[k*units + j];
                    }
                }
            }
            
    }
};
template <typename Device, typename T>
class DenseamOp: public OpKernel{
    public:
        explicit DenseamOp(OpKernelConstruction* context): OpKernel(context),
        mul_lut_(context) {
        }
        void Compute(OpKernelContext* context) override {
            const Tensor& input = context->input(0);
            const Tensor& weights = context->input(1);
            const TensorShape& input_shape = input.shape();
            const TensorShape& weights_shape = weights.shape();
            DCHECK_EQ(input_shape.dims(), 2);
            DCHECK_EQ(weights_shape.dims(), 2);
            
            const int batch = input_shape.dim_size(0);
           
            const int input_width = input_shape.dim_size(1);
           
            const int units = weights_shape.dim_size(1);
           
            DCHECK_EQ(input_width, weights_shape.dim_size(0));
           
            TensorShape output_shape;
            output_shape.AddDim(batch);
            output_shape.AddDim(units);
                    
            Tensor* output = NULL;
            OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
            
            // get the corresponding Eigen tensors for data access
            auto input_tensor = input.flat<T>().data();
            auto weights_tensor = weights.flat<T>().data();
            auto output_tensor = output->flat<T>().data();
            DenseamFunctor<Device, T>()(
                    context->eigen_device<Device>(),
                    input_tensor,
                    weights_tensor,
                    output_tensor,
                    batch,
                    units,
                    input_width,
                    mul_lut_
                    );
        }

  private:
  approx_mul_lut<Device> mul_lut_;
  TF_DISALLOW_COPY_AND_ASSIGN(DenseamOp);
};

// Register the CPU kernels.
#define REGISTER_CPU_DENSEAM(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Denseam").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DenseamOp<CPUDevice, T>);
REGISTER_CPU_DENSEAM(float);
REGISTER_CPU_DENSEAM(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU_DENSEAM(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template class DenseamFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Denseam").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DenseamOp<GPUDevice, T>);
REGISTER_GPU_DENSEAM(float);
REGISTER_GPU_DENSEAM(int32);
#endif  // GOOGLE_CUDA

template <typename T>
struct DenseamWeightGradFunctor<CPUDevice, T>{
    void operator()(const CPUDevice& d, const T* input, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<CPUDevice>& mul_lut
            ){
            for(int i = 0; i < batch; i++){
                for(int j = 0; j < units; j++){
                    for(int k = 0; k < input_width; k++){
                        output[k*units + j] += input[i*input_width + k] * grads[i*units + j];
                    }
                }
            }
    }
};
template <typename T>
struct DenseamInputGradFunctor<CPUDevice, T>{
    void operator()(const CPUDevice& d, const T* weight, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<CPUDevice>& mul_lut
            ){
            for(int i = 0; i < batch; i++)
            for(int i = 0; i < batch; i++){
                for(int j = 0; j < units; j++){
                    for(int k = 0; k < input_width; k++){
                        output[i*input_width +k] += weight[k*units + j] * grads[i*units + j];
                    }
                }
            }
    }
};

REGISTER_OP("DenseamGrad")
  .Input("grad: T")
  .Input("input: T")
  .Input("weights: T")
  .Output("grad_input: T")
  .Output("grad_weights: T")
  .Attr("T: {float, int32}")
  .Attr("mant_mul_lut: string");
template<typename Device, typename T>
class DenseamGradOp: public OpKernel {
public:
  explicit DenseamGradOp(OpKernelConstruction* context) : OpKernel(context),
    mul_lut_(context) {
  }
  void Compute(OpKernelContext* context) override {

    const Tensor& grad_t = context->input(0);
    const Tensor& input_t = context->input(1);
    const Tensor& weights_t = context->input(2);
    
    
    TensorShape grad_shape = grad_t.shape();
    TensorShape input_shape = input_t.shape();
    TensorShape weights_shape = weights_t.shape();
    Tensor* grad_input_t = NULL;
    Tensor* grad_weights_t = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input_t));
    OP_REQUIRES_OK(context, context->allocate_output(1, weights_shape, &grad_weights_t));
    
    int input_width= input_shape.dim_size(1);  //Number of values in each sample
    int batch = input_shape.dim_size(0); //Number of samples in batch
    int units = weights_shape.dim_size(1); //Number of units
    
    auto grad = grad_t.flat<T>().data();
    auto input = input_t.flat<T>().data();
    auto weights = weights_t.flat<T>().data();
    auto grad_input = grad_input_t->template flat<T>().data();
    auto grad_weights = grad_weights_t->template flat<T>().data();
    DenseamWeightGradFunctor<Device, T>()(
            context->eigen_device<Device>(),    
            input,
            grad,
            grad_weights,
            batch,
            units,
            input_width,
            mul_lut_
            );
    DenseamInputGradFunctor<Device, T>()(
            context->eigen_device<Device>(),    
            weights,
            grad,
            grad_input,
            batch,
            units,
            input_width,
            mul_lut_
            );
  }
  private:
  approx_mul_lut<Device> mul_lut_;
  TF_DISALLOW_COPY_AND_ASSIGN(DenseamGradOp);
};
// Register the CPU kernels.
#define REGISTER_CPU_DENSEAMGRAD(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("DenseamGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DenseamGradOp<CPUDevice, T>);
REGISTER_CPU_DENSEAMGRAD(float);
REGISTER_CPU_DENSEAMGRAD(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU_DENSEAMGRAD(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template class DenseamWeightGradFunctor<GPUDevice, T>;            \
  extern template class DenseamInputGradFunctor<GPUDevice, T>;            \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("DenseamGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      DenseamGradOp<GPUDevice, T>);
REGISTER_GPU_DENSEAMGRAD(float);
REGISTER_GPU_DENSEAMGRAD(int32);
#endif  // GOOGLE_CUDA
