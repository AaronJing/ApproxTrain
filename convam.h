#ifndef CONVAM_H_
#define CONVAM_H_

#include <unsupported/Eigen/CXX11/Tensor>
#include "approx_mul_lut.h"
//#include <tensorflow/core/framework/op_kernel.h>
//#include <fstream>
//typedef unsigned long long cudaTextureObject_t;
//class approx_mul_lut_base {
    //public:
        //explicit approx_mul_lut_base(tensorflow::OpKernelConstruction* context) :
            //mant_mul_lut_{0}, exp_mul_lut_{0} {
            //load_lut_binary();
        //}
        //virtual ~approx_mul_lut_base() = default;
        //// same for both CPU and GPU
        //auto load_lut_binary() -> void {
            //// open mant mul file
            //std::ifstream file("mbmmantmul16.bin", std::ios::in | std::ios::binary);
            //if(file.fail()) {
                //std::cerr << "file mbmmantmul16.bin failed" << std::endl;
                //exit(1);
            //} 
            //if(!file.is_open()) { 
                //std::cerr << "file mbmmantmul16.bin open failed" << std::endl;
                //exit(1);
            //}
            //mant_mul_lut_.resize(128*128);
            //file.read(
                    //reinterpret_cast<char *>(mant_mul_lut_.data()), 
                    //mant_mul_lut_.size() * sizeof(uint32_t)
                //);
            //// open exponent file
            //std::ifstream exp_file("exp.bin", std::ios::in|std::ios::binary);
            //if(exp_file.fail()) {
                //std::cerr << "file exp.bin failed" << std::endl;
                //exit(1);
            //} 
            //if(!exp_file.is_open()) { 
                //std::cerr << "file exp.bin open failed" << std::endl;
                //exit(1);
            //}
            //exp_mul_lut_.resize(2*2*256);
            //exp_file.read(
                    //reinterpret_cast<char *>(exp_mul_lut_.data()),
                    //exp_mul_lut_.size() * sizeof(uint32_t)
                    //);
        //}
        //auto get_mant_mul_lut_text_() -> cudaTextureObject_t& {
            //return mant_mul_lut_text_;
        //}
        //auto get_exp_mul_lut_text_() -> cudaTextureObject_t& {
            //return exp_mul_lut_text_;
        //}
    //protected:
        //std::vector<uint32_t> mant_mul_lut_;
        //std::vector<uint32_t> exp_mul_lut_;
        //uint32_t* mant_mul_lut_cuda_;
        //uint32_t* exp_mul_lut_cuda_;
        //cudaTextureObject_t mant_mul_lut_text_;
        //cudaTextureObject_t exp_mul_lut_text_;
//};
//template <typename Device>
//class approx_mul_lut : public approx_mul_lut_base {
    //public:
        //explicit approx_mul_lut(tensorflow::OpKernelConstruction *context);
        //~approx_mul_lut();
//};
template <typename Device, typename T>
struct ConvamFunctor {
  void operator()(const Device& d, const T* input_data, T* output_data,
            const int batch, const int out_rows, const int out_cols,
            const int out_depth, const int stride_cols, const int stride_rows,
            const int filter_left_offset, const int filter_top_offset,
            const int filter_rows, const int filter_cols, const int in_depth,
            const int input_cols, const int input_rows, const T* filter,
            T* im2col, const int padding,
            approx_mul_lut<Device>& mul_lut
          );
};

template <typename Device, typename T>
struct ConvamInputGradFunctor {
  void operator()(const Device& d, const T* grad, T* im2col,
          const int hole_grad_width, const int hole_grad_height,
          const int pad_top, const int pad_left, const T* filter, T* rsfilter,
          const int filter_rows, const int filter_cols, const int out_depth,
          const int stride_rows, const int stride_cols, const int batch,
          const int input_rows, const int input_cols, const int in_depth,
          T* output, const int out_rows, const int out_cols, 
          approx_mul_lut<Device>& mul_lut
          );
};

template <typename Device, typename T>
struct ConvamFilterGradFunctor{
  void operator()(const Device& d, const T* input, const T* grad, T* im2col,
          const int input_rows, const int input_cols, const int batch,
          const int in_depth, const int out_cols, const int out_rows,
          const int out_depth, const int filter_left_offset,
          const int filter_top_offset, const int stride_rows,
          const int stride_cols, const int filter_cols, const int filter_rows,
          T* output, approx_mul_lut<Device>& mul_lut
          );
};
#ifdef GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
template <typename T>
struct ConvamFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, const T* input_data,
            T* output_data, const int batch, const int out_rows, const int out_cols,
            const int out_depth, const int stride_cols, const int stride_rows,
            const int filter_left_offset, const int filter_top_offset,
            const int filter_rows, const int filter_cols, const int in_depth,
            const int input_cols, const int input_rows, const T* filter,
            T* im2col, const int padding, 
            approx_mul_lut<Eigen::GpuDevice>& mul_lut
          );
};

template <typename T>
struct ConvamInputGradFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, const T* grad, T* im2col,
          const int hole_grad_width, const int hole_grad_height,
          const int pad_top, const int pad_left, const T* filter, T* rsfilter,
          const int filter_rows, const int filter_cols, const int out_depth,
          const int stride_rows, const int stride_cols, const int batch,
          const int input_rows, const int input_cols, const int in_depth,
          T* output, const int out_rows, const int out_cols,
          approx_mul_lut<Eigen::GpuDevice>& mul_lut
          );
};
template <typename T>
struct ConvamFilterGradFunctor<Eigen::GpuDevice, T>{
  void operator()(const Eigen::GpuDevice& d, const T* input, const T* grad,
          T* im2col, const int input_rows, const int input_cols,
          const int batch, const int in_depth, const int out_cols,
          const int out_rows,const int out_depth, const int filter_left_offset,
          const int filter_top_offset, const int stride_rows,
          const int stride_cols, const int filter_cols, const int filter_rows,
          T* output, approx_mul_lut<Eigen::GpuDevice>& mul_lut
          );
};


#endif

#endif
