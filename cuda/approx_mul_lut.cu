#define EIGEN_USE_GPU
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iterator>
#include "error.cuh"
#include "approx_mul_lut.h"
using namespace tensorflow;

template<>
class approx_mul_lut<Eigen::GpuDevice> : public approx_mul_lut_base {
    public:
        explicit approx_mul_lut(tensorflow::OpKernelConstruction* context);
        ~approx_mul_lut();
        auto get_mant_mul_lut_text_() -> cudaTextureObject_t& {
            return mant_mul_lut_text_;
        }
        auto get_mant_mul_lut_() -> uint32_t* {
            return mant_mul_lut_cuda_;
        }
        auto get_exp_mul_lut_text_() -> cudaTextureObject_t& {
            return exp_mul_lut_text_;
        }

};
approx_mul_lut<Eigen::GpuDevice>::approx_mul_lut(OpKernelConstruction * context):
            approx_mul_lut_base(context){

    gpuErrchk(cudaMalloc(&mant_mul_lut_cuda_, 
            mant_mul_lut_.size() * sizeof(uint32_t)));
    gpuErrchk(cudaMemcpy(mant_mul_lut_cuda_, mant_mul_lut_.data(),
            mant_mul_lut_.size()*sizeof(uint32_t), 
            cudaMemcpyHostToDevice));
    cudaResourceDesc mant_mul_lut_res_desc;
    memset(&mant_mul_lut_res_desc, 0, sizeof(cudaResourceDesc));
    mant_mul_lut_res_desc.resType = cudaResourceTypeLinear;
    mant_mul_lut_res_desc.res.linear.devPtr = mant_mul_lut_cuda_;
    mant_mul_lut_res_desc.res.linear.desc.f = 
        cudaChannelFormatKindUnsigned;
    mant_mul_lut_res_desc.res.linear.desc.x = 32;
    mant_mul_lut_res_desc.res.linear.sizeInBytes = 
        mant_mul_lut_.size() * sizeof(uint32_t);
    
    cudaTextureDesc mant_mul_text_desc;
    memset(&mant_mul_text_desc, 0, sizeof(cudaTextureDesc));
    mant_mul_text_desc.readMode = cudaReadModeElementType;
        
    gpuErrchk(cudaCreateTextureObject(&mant_mul_lut_text_, &mant_mul_lut_res_desc, 
            &mant_mul_text_desc, nullptr));                

    gpuErrchk(cudaMalloc(&exp_mul_lut_cuda_, 
            exp_mul_lut_.size() * sizeof(uint32_t)));
    gpuErrchk(cudaMemcpy(exp_mul_lut_cuda_, exp_mul_lut_.data(),
            exp_mul_lut_.size()*sizeof(uint32_t), 
            cudaMemcpyHostToDevice));
    cudaResourceDesc exp_mul_lut_res_desc;
    memset(&exp_mul_lut_res_desc, 0, sizeof(cudaResourceDesc));
    exp_mul_lut_res_desc.resType = cudaResourceTypeLinear;
    exp_mul_lut_res_desc.res.linear.devPtr = exp_mul_lut_cuda_;
    exp_mul_lut_res_desc.res.linear.desc.f = 
        cudaChannelFormatKindUnsigned;
    exp_mul_lut_res_desc.res.linear.desc.x = 32;
    exp_mul_lut_res_desc.res.linear.sizeInBytes = 
        exp_mul_lut_.size() * sizeof(uint32_t);
    
    cudaTextureDesc exp_mul_text_desc;
    memset(&exp_mul_text_desc, 0, sizeof(cudaTextureDesc));
    exp_mul_text_desc.readMode = cudaReadModeElementType;
        
    gpuErrchk(cudaCreateTextureObject(&exp_mul_lut_text_, &exp_mul_lut_res_desc, 
            &exp_mul_text_desc, nullptr));                
};

approx_mul_lut<Eigen::GpuDevice>::~approx_mul_lut(){
    cudaDestroyTextureObject(mant_mul_lut_text_);
    cudaFree(mant_mul_lut_cuda_);
    cudaDestroyTextureObject(exp_mul_lut_text_);
    cudaFree(exp_mul_lut_cuda_);
};
