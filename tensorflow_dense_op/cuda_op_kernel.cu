#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>

#include "gpu_kernel_helper.h"
#include "error.cuh"
using GPUDevice = Eigen::GpuDevice;
template <typename T>
__global__ void densekernel(
    const T* inputs,
    const T* weights,
    const int batch, 
    const int units, 
    const int input_feature_width, 
    T* output) 
{ 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if(i < units*batch)
    {
        int i_unit = i % units ;
        int i_sample = i / units;
        output[i] = 0;
        for (int i_input = 0; i_input < input_feature_width; i_input++)
        {
          output[i] += inputs[i_sample*input_feature_width+i_input] * weights[i_input*units+i_unit];
        }  
    }
}

template <typename T>
void densekernellauncher(
        const GPUDevice &d,
        const T* inputs, 
        const T* weights,
        const int batch, 
        const int units, 
        const int input_feature_width,
        T* output) 
{
    densekernel<T><<<batch, units, 0, d.stream()>>>(inputs, weights, biases, batch, units, input_feature_width, output);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}


template <typename T>
__global__ void inputgrad(
    const T* grads,
    const T* weights,
    const int input_feature_width, 
    const int batch, 
    const int units, 
    T* grad_inputs) 
{ 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if(i < batch*input_feature_width)
    {
        int i_input = i % input_feature_width;
        int i_sample = i / input_feature_width ;
        grad_inputs[i] = T(0);

        for (int i_unit = 0; i_unit < units; i_unit++)
        {
			grad_inputs[i_sample*input_feature_width+i_input] += weights[i_input*units+ i_unit]*grads[i_sample*units+i_unit];
        }  
    }
}
template <typename T>
void inputgradkernellauncher(
    const GPUDevice &d,
    const T* grads, 
    const T* weights, 
    const int input_feature_width, 
    const int batch, 
    const int units, 
    T* grad_inputs)
{
    inputgrad<T><<<batch,input_feature_width, 0, d.stream()>>>(grads, weights, input_feature_width, batch, units, grad_inputs);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

// Weights gradient
template <typename T>
__global__ void weightsgrad(
    const T* grads,
    const T* inputs,
    const int input_feature_width, 
    const int batch, 
    const int units, 
    T* grad_weights) 
{ 
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if(i < units*input_feature_width)
    {
        int i_unit = i % units ;
        int i_input = i / units;
        grad_weights[i] = T(0);
        for (int i_sample = 0; i_sample < batch; i_sample++)
        {
           grad_weights[i] += inputs[input_feature_width*i_sample+i_input]*grads[i_sample*units+i_unit];
        }  
    }
}
template <typename T>
void weightsgradkernellauncher(
    const GPUDevice &d,
    const T* grads, 
    const T* inputs, 
    const int input_feature_width, 
    const int batch, 
    const int units, 
    T* grad_weights)
{
    weightskernel<T><<<units,input_feature_width>>>(grads, inputs, input_feature_width, batch, units, grad_weights);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

template <typename T>
struct DenseamFunctor<Eigen::GpuDevice, T>::operator()(
        const Eigen::GpuDevice& d, const T* inputs, const T* weights, 
        const int batch, const int units, const int input_feature_width,
        T* output
        ) {
    densekernellauncher<T>(d, inputs, weights, batch, units, input_feature_width,
            output);
};

template <typename T>
struct DenseamInputGradFunctor<Eigen::GpuDevice, T>::operator()(
        const Eigen:GpuDevice& d, const T* grads, const T* weights, 
        const int input_feature_width, const int batch, const int units, 
        T* grad_inputs 
        ) {
    inputgradkernellauncher<T>(d, grads, weights, input_feature_width, batch, units,
            grad_inputs);
};

template <typename T>
struct DenseamWeightGradFunctor<Eigen::GpuDevice, T>::operator()(
        const Eigen::GpuDevice& d, const T* grads, const T* inputs, 
        const int input_feature_width, const int batch, const int units, 
        T* grad_weights 
        ){
    weightsgradkernellauncher<T>(d, grads, inputs, input_feature_width, batch
            units, grad_weights);
};
