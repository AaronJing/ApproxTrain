#ifndef DENSEAM_H_
#define DENSEAM_H_

#include <unsupported/Eigen/CXX11/Tensor>
#include "approx_mul_lut.h"

template <typename Device, typename T>
struct DenseamFunctor {
    void operator()(const Device& d, const T* inputs, const T* weights, T* output,
            const int batch, const int units, const int input_width,
            approx_mul_lut<Device>& mul_lut
            );
};

template <typename Device, typename T>
struct DenseamWeightGradFunctor{
    void operator()(const Device& d, const T* input, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<Device>& mul_lut
            );
};
template <typename Device, typename T>
struct DenseamInputGradFunctor{
    void operator()(const Device& d, const T* weight, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<Device>& mul_lut
            ); 
};


#if GOOGLE_CUDA
template <typename T>
struct DenseamFunctor<Eigen::GpuDevice, T> {
    void operator()(const Eigen::GpuDevice& d, const T* inputs, const T* weights, 
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<Eigen::GpuDevice>& mul_lut
            );
};

template <typename T>
struct DenseamWeightGradFunctor<Eigen::GpuDevice, T>{
    void operator()(const Eigen::GpuDevice& d, const T* input, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<Eigen::GpuDevice>& mul_lut
            );
};
template <typename T>
struct DenseamInputGradFunctor<Eigen::GpuDevice, T>{
    void operator()(const Eigen::GpuDevice& d, const T* weight, const T* grads,
            T* output, const int batch, const int units, const int input_width,
            approx_mul_lut<Eigen::GpuDevice>& mul_lut
            ); 
};
#endif
#endif
