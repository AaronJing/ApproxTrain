#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

/*
    Register Dense operation
*/

REGISTER_OP("Dense")
  .Input("input: float")
  .Input("weights: float")
  .Input("biases: float")
  .Output("output: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {

    shape_inference::ShapeHandle input_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));

    shape_inference::ShapeHandle weight_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &weight_shape));
    
    shape_inference::ShapeHandle biases_shape;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &biases_shape));
    
    shape_inference::DimensionHandle samples = c->Dim(input_shape, 0);
    shape_inference::DimensionHandle units = c->Dim(weight_shape, 1);
    
    c->set_output(0, c->Matrix(samples, units));

    return Status::OK();
  });

/*
    Dense Operation CPU
*/

class DenseOpCPU : public OpKernel {
public:
  explicit DenseOpCPU(OpKernelConstruction* context) : OpKernel(context) {
  }
  
  void Compute(OpKernelContext* context) override {
    //printf("DenseOpCPU\n");

    // get the input tensor
    const Tensor& input = context->input(0);
    
    // get the weight tensor
    const Tensor& weights = context->input(1);
      
    // get the bias tensor
    const Tensor& biases = context->input(2);
    
    // check shapes of input and weights
    const TensorShape& input_shape = input.shape();
    const TensorShape& weights_shape = weights.shape();
    const TensorShape& biases_shape = biases.shape();
    
    //Check that inputs are two dimensional
    DCHECK_EQ(input_shape.dims(), 2);
    DCHECK_EQ(weights_shape.dims(), 2);
    DCHECK_EQ(biases_shape.dims(), 2);
    
    const int batch_samples = input_shape.dim_size(0);
    //printf("batch_samples %d\n", batch_samples);

    const int input_feature_width = input_shape.dim_size(1);
    //printf("input_feature_width %d\n", input_feature_width);

    const int units = weights_shape.dim_size(1);
    //printf("units %d\n", units);

    //Check input width matches weights height 
    DCHECK_EQ(input_feature_width, weights_shape.dim_size(0));
    //Check weights width match bias width 
    DCHECK_EQ(weights_shape.dim_size(1), biases_shape.dim_size(1));

    // create output shape
    TensorShape output_shape;
    //printf("batch_samples: %d\n", batch_samples);
    //printf("units: %d\n", units);

    output_shape.AddDim(batch_samples);
    output_shape.AddDim(units);
            
    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    // get the corresponding Eigen tensors for data access
    auto input_tensor = input.matrix<float>();
    auto weights_tensor = weights.matrix<float>();
    auto biases_tensor = biases.matrix<float>();
    auto output_tensor = output->matrix<float>();
    
    for (int ix_sample = 0; ix_sample < batch_samples; ix_sample++) {
      for (int ix_unit = 0; ix_unit < units; ix_unit++) {
        output_tensor(ix_sample, ix_unit) = 0;
        for (int ix_input = 0; ix_input < input_feature_width; ix_input++) {
          output_tensor(ix_sample, ix_unit) += input_tensor(ix_sample, ix_input) * weights_tensor(ix_input, ix_unit );
        }  
        output_tensor(ix_sample, ix_unit) += biases_tensor(0, ix_unit);
      }
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Dense").Device(DEVICE_CPU), DenseOpCPU);

/*
    Dense Operation GPU
*/

void DenseKernelLauncher(
        const float* inputs, 
        const float* weights,
        const float* biases,
        const int batch_samples, 
        const int units, 
        const int input_feature_width,
        float* output);



class DenseOpGPU : public OpKernel {
public:
  explicit DenseOpGPU(OpKernelConstruction* context) : OpKernel(context) {
  }
  
  void Compute(OpKernelContext* context) override {
    //printf("DenseOpGPU\n");

    // get the input tensor
    const Tensor& input = context->input(0);
    
    // get the weight tensor
    const Tensor& weights = context->input(1);
      
    // get the bias tensor
    const Tensor& biases = context->input(2);
    
    // check shapes of input and weights
    const TensorShape& input_shape = input.shape();
    const TensorShape& weights_shape = weights.shape();
    const TensorShape& biases_shape = biases.shape();
    
    //Check that inputs are two dimensional
    DCHECK_EQ(input_shape.dims(), 2);
    DCHECK_EQ(weights_shape.dims(), 2);
    DCHECK_EQ(biases_shape.dims(), 2);
    
    const int batch_samples = input_shape.dim_size(0);
    //printf("batch_samples %d\n", batch_samples);

    const int input_feature_width = input_shape.dim_size(1);
    //printf("input_feature_width %d\n", input_feature_width);

    const int units = weights_shape.dim_size(1);
    //printf("units %d\n", units);

    //Check input width matches weights height 
    DCHECK_EQ(input_feature_width, weights_shape.dim_size(0));
    //Check weights width match bias width 
    DCHECK_EQ(weights_shape.dim_size(1), biases_shape.dim_size(1));

    // create output shape
    TensorShape output_shape;
    //printf("batch_samples: %d\n", batch_samples);
    //printf("units: %d\n", units);

    output_shape.AddDim(batch_samples);
    output_shape.AddDim(units);
            
    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    auto f_input = input.flat<float>();
    auto f_weights = weights.flat<float>();
    auto f_biases = biases.flat<float>();
    auto f_output = output->template flat<float>();

    DenseKernelLauncher(
            f_input.data(), 
            f_weights.data(),
            f_biases.data(),
            batch_samples, 
            units, 
            input_feature_width,
            f_output.data()
        );
  }
};

REGISTER_KERNEL_BUILDER(Name("Dense").Device(DEVICE_GPU), DenseOpGPU);



/*
    DenseGrad Operation CPU
*/


REGISTER_OP("DenseGrad")
  .Input("grad: float")
  .Input("input: float")
  .Input("weights: float")
  .Input("biases: float")
  .Output("grad_input: float")
  .Output("grad_weights: float")
  .Output("grad_biases: float");

class DenseGradOpCPU : public OpKernel {
public:
  explicit DenseGradOpCPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
    //printf("DenseGradOpCPU\n");
  
    DCHECK_EQ(4, context->num_inputs());

    const Tensor& grad = context->input(0);
    
    const Tensor& input = context->input(1);
    
    const Tensor& weights = context->input(2);
    
    const Tensor& biases = context->input(3);
    
    TensorShape grad_shape = grad.shape();
    TensorShape input_shape = input.shape();
    TensorShape weights_shape = weights.shape();
    TensorShape biases_shape = biases.shape();
    
    // create output tensors
    Tensor* grad_input = NULL;
    Tensor* grad_weights = NULL;
    Tensor* grad_biases = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));
    OP_REQUIRES_OK(context, context->allocate_output(1, weights_shape, &grad_weights));
    OP_REQUIRES_OK(context, context->allocate_output(2, biases_shape, &grad_biases));
    
    // get the Eigen tensors for data access
    auto grad_tensor = grad.matrix<float>();
    auto weights_tensor = weights.matrix<float>();
    auto input_tensor = input.matrix<float>();
    
    auto grad_input_tensor = grad_input->matrix<float>();
    auto grad_weights_tensor = grad_weights->matrix<float>();
    auto grad_biases_tensor = grad_biases->matrix<float>();

    int input_feature_width = input_shape.dim_size(1);  //Number of values in each sample
    int batch_samples = input_shape.dim_size(0); //Number of samples in batch
    int units = weights_shape.dim_size(1); //Number of units

    
    for (int x = 0; x < units; x++) 
    {
        grad_biases_tensor(0, x) = 0.0;
    }
    for (int x = 0; x < units; x++) //unit index 
    {
        for (int y = 0; y < input_feature_width; y++) //input feature index
        {
            grad_weights_tensor(y, x) = 0.0;
        }
    }
    for (int x = 0; x < input_feature_width; x++) 
    {
        for (int y = 0; y < batch_samples; y++) 
        {
            grad_input_tensor(y, x) = 0.0;
        }
    }
    for (int ix_sample = 0; ix_sample < batch_samples; ix_sample++) {
      for (int ix_unit = 0; ix_unit < units; ix_unit++) {
        for (int ix_input = 0; ix_input < input_feature_width; ix_input++) {
            grad_input_tensor(ix_sample, ix_input) += weights_tensor(ix_input, ix_unit )*grad_tensor(ix_sample, ix_unit);
            grad_weights_tensor(ix_input, ix_unit ) += input_tensor(ix_sample, ix_input)*grad_tensor(ix_sample, ix_unit);
        }  
        grad_biases_tensor(0, ix_unit) += grad_tensor(ix_sample, ix_unit);
      }
    }    
    
  }
};

REGISTER_KERNEL_BUILDER(Name("DenseGrad").Device(DEVICE_CPU), DenseGradOpCPU);


/*
    DenseGrad Operation GPU
*/

void InputGradKernelLauncher(const float* grads, const float* weights, const int input_feature_width, const int batch_samples, const int units, float* grad_inputs);
void WeightsGradKernelLauncher(const float* grads, const float* inputs, const int input_feature_width, const int batch_samples, const int units, float* grad_weights);
void BiasGradKernelLauncher(const float* grads, const int input_feature_width, const int batch_samples, const int units, float* grad_biases);
  
class DenseGradOpGPU : public OpKernel {
public:
  explicit DenseGradOpGPU(OpKernelConstruction* context) : OpKernel(context) {
    
  }
  
  void Compute(OpKernelContext* context) override {
  
    //printf("DenseGradOpGPU\n");
  
    DCHECK_EQ(4, context->num_inputs());

    const Tensor& grad = context->input(0);
    
    const Tensor& input = context->input(1);
    
    const Tensor& weights = context->input(2);
    
    const Tensor& biases = context->input(3);
    
    TensorShape grad_shape = grad.shape();
    TensorShape input_shape = input.shape();
    TensorShape weights_shape = weights.shape();
    TensorShape biases_shape = biases.shape();
    
    // create output tensors
    Tensor* grad_input = NULL;
    Tensor* grad_weights = NULL;
    Tensor* grad_biases = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input));
    OP_REQUIRES_OK(context, context->allocate_output(1, weights_shape, &grad_weights));
    OP_REQUIRES_OK(context, context->allocate_output(2, biases_shape, &grad_biases));
    
    int input_feature_width = input_shape.dim_size(1);  //Number of values in each sample
    int batch_samples = input_shape.dim_size(0); //Number of samples in batch
    int units = weights_shape.dim_size(1); //Number of units
    
    auto f_grad = grad.flat<float>();
    auto f_input = input.flat<float>();
    auto f_weights = weights.flat<float>();
    auto f_biases = biases.flat<float>();
    auto f_grad_input = grad_input->template flat<float>();
    auto f_grad_weights = grad_weights->template flat<float>();
    auto f_grad_biases = grad_biases->template flat<float>();
    
    InputGradKernelLauncher(f_grad.data(), f_weights.data(), input_feature_width,batch_samples,units,f_grad_input.data());
    WeightsGradKernelLauncher(f_grad.data(), f_input.data(), input_feature_width,batch_samples,units,f_grad_weights.data());
    BiasGradKernelLauncher(f_grad.data(), input_feature_width,batch_samples,units,f_grad_biases.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("DenseGrad").Device(DEVICE_GPU), DenseGradOpGPU);

