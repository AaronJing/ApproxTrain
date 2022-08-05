#define EIGEN_USE_GPU

#include "gpu_kernel_helper.h"
#include "error.cuh"
#include "denseam.h"
#include "approx_mul_lut.h"

using namespace tensorflow;
using GpuDevice = Eigen::GpuDevice;
#ifdef AMSIMULATOR
   #define MULTIPLY(a,b) AMsimulator((a), (b), lut, mant_mask, a_shift, b_shift, mant_bitwidth);
   #include "AMsimulator.inl"
#else
   #define MULTIPLY(a,b) ((a)*(b));
#endif


template <typename T>
__global__ void DenseamKernel(
    const T* inputs,
    const T* weights,
    const int batch, 
    const int units, 
    const int input_width, 
    T* output, 
    cudaTextureObject_t lut,
    const uint32_t mant_mask,
    const uint8_t a_shift,
    const uint8_t b_shift, const uint8_t mant_bitwidth
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*batch)
    {
        int ix_unit = ix % units ;
        int ix_sample = ix / units;
        output[ix] = T(0);
        for (int ix_input = 0; ix_input < input_width; ix_input++)
        {
          output[ix] += MULTIPLY(inputs[ix_sample*input_width+ix_input], weights[ix_input*units+ix_unit]);
        }  
    }
};
template <typename T>
__global__ void DenseamWeightsKernel(
    const T* grads,
    const T* inputs,
    const int input_width, 
    const int batch, 
    const int units, 
    T* grad_weights,
    cudaTextureObject_t lut,
    const uint32_t mant_mask,
    const uint8_t a_shift,
    const uint8_t b_shift, const uint8_t mant_bitwidth
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < units*input_width)
    {
        int ix_unit = ix % units ;
        int ix_input = ix / units;
        grad_weights[ix] = T(0);
        for (int ix_sample = 0; ix_sample < batch; ix_sample++)
        {
           grad_weights[ix] += MULTIPLY(inputs[input_width*ix_sample+ix_input], grads[ix_sample*units+ix_unit]);
        }  
    }
};
template <typename T>
__global__ void DenseamInputKernel(
    const T* grads,
    const T* weights,
    const int input_width, 
    const int batch, 
    const int units, 
    T* grad_inputs, 
    cudaTextureObject_t lut,
    const uint32_t mant_mask,
    const uint8_t a_shift,
    const uint8_t b_shift, const uint8_t mant_bitwidth
    ) 
{ 
    unsigned int ix = blockIdx.x * blockDim.x + threadIdx.x; 
    if(ix < batch *input_width)
    {
        int ix_input = ix % input_width;
        int ix_sample = ix / input_width ;
        grad_inputs[ix] = T(0);

        for (int ix_unit = 0; ix_unit < units; ix_unit++)
        {
			grad_inputs[ix_sample*input_width+ix_input] += MULTIPLY(weights[ix_input*units+ ix_unit], grads[ix_sample*units+ix_unit]);
        }  
    }
};
template <typename T>
void DenseamFunctor<GpuDevice, T>::operator()(
        const GpuDevice& d, const T* inputs, const T* weights, T* output,
        const int batch, const int units, const int input_width,
        approx_mul_lut<GpuDevice>& mul_lut )
{ 
        const uint32_t mant_mask = mul_lut.get_mant_mask_();
        const uint8_t a_shift = mul_lut.get_a_shift_();
        const uint8_t b_shift = mul_lut.get_b_shift_();
        const uint8_t mant_bitwidth = mul_lut.get_mant_width_();
        unsigned blocksize = 1024;
        unsigned gridsize = (batch*units+blocksize -1)/blocksize;
        DenseamKernel<T><<<gridsize, blocksize, 0, d.stream()>>>(inputs, weights, batch, units, input_width, output, mul_lut.get_mant_mul_lut_text_(), mant_mask, a_shift, b_shift, mant_bitwidth);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
}

template <typename T>
void DenseamWeightGradFunctor<GpuDevice, T>::operator()
    (const GpuDevice& d, const T* input, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<GpuDevice>& mul_lut ) 
            {
    const uint32_t mant_mask = mul_lut.get_mant_mask_();
    const uint8_t a_shift = mul_lut.get_a_shift_();
    const uint8_t b_shift = mul_lut.get_b_shift_();
    const uint8_t mant_bitwidth = mul_lut.get_mant_width_();
    unsigned blocksize = 1024;
    unsigned gridsize = (units*input_width+blocksize -1)/blocksize;
    DenseamWeightsKernel<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, input, input_width, batch, units, output, mul_lut.get_mant_mul_lut_text_(), mant_mask, a_shift, b_shift, mant_bitwidth);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
template <typename T>
void DenseamInputGradFunctor<GpuDevice, T>::operator()
    (const GpuDevice& d, const T* weight, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<GpuDevice>& mul_lut
            ){
    const uint32_t mant_mask = mul_lut.get_mant_mask_();
    const uint8_t a_shift = mul_lut.get_a_shift_();
    const uint8_t b_shift = mul_lut.get_b_shift_();
    const uint8_t mant_bitwidth = mul_lut.get_mant_width_();
    unsigned blocksize = 1024;
    unsigned gridsize = (batch*input_width+blocksize -1)/blocksize;
    DenseamInputKernel<T><<<gridsize, blocksize, 0, d.stream()>>>(grads, weight, input_width, batch, units, output, mul_lut.get_mant_mul_lut_text_(), mant_mask, a_shift, b_shift, mant_bitwidth);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}

template struct DenseamFunctor<GpuDevice, float>;
template struct DenseamFunctor<GpuDevice, int32>;
template struct DenseamInputGradFunctor<GpuDevice, float>;
template struct DenseamInputGradFunctor<GpuDevice, int32>;
template struct DenseamWeightGradFunctor<GpuDevice, float>;
template struct DenseamWeightGradFunctor<GpuDevice, int32>;
