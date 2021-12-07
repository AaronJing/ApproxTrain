#ifndef DENSEAM_H_
#define DENSEAM_H_

#include <unsupported/Eigen/CXX11/Tensor>

template <typename Device, typename T>
struct DenseamFunctor {
    void operator()(const Device& d, const T* inputs, const T* weights, 
            const int batch, const int units, const int input_feature_width,
            T* output
            );
};

template <typename Device, typename T>
struct DenseamInputFunctor {
    void operator()(const Device& d, const T* grads, const T* weights, 
            const int input_feature_width, const int batch, const int units, 
            T* grad_inputs 
            );
};

template <typename Device, typename T>
struct DenseamFilterFunctor{
    void operator()(const Device& d, const T* grads, const T* inputs, 
            const int input_feature_width, const int batch, const int units, 
            T* grad_weights 
            );
};
#if GOOGLE_CUDA
template <typename T>
struct DenseamFunctor<Eigen::GpuDevice, T> {
    void operator()(const Eigen::GpuDevice& d, const T* inputs, const T* weights, 
            const int batch, const int units, const int input_feature_width,
            T* output
            );
};

template <typename T>
struct DenseamInputGradFunctor<Eigen::GpuDevice, T> {
    void operator()(const Eigen:GpuDevice& d, const T* grads, const T* weights, 
            const int input_feature_width, const int batch, const int units, 
            T* grad_inputs 
            );
};

template <typename T>
struct DenseamWeightGradFunctor<Eigen::GpuDevice, T>{
    void operator()(const Eigen::GpuDevice& d, const T* grads, const T* inputs, 
            const int input_feature_width, const int batch, const int units, 
            T* grad_weights 
            );
};
#endif
#endif
