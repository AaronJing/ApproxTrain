#define EIGEN_USE_GPU
//#include <cuda.h>
//#include <stdio.h>
//#include <fstream>
//#include <assert.h>
//#include <cuda_fp16.h>
//#include <iostream>
//#include <chrono>
//#include <sys/time.h>

#include "gpu_kernel_helper.h"
#include "error.cuh"
#include "gemm.cuh"
#include "reverseNswapdim23.cuh"
#include "convam.h"
#include "approx_mul_lut.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <iterator>

using namespace tensorflow;
//using namespace std;
#define THREADS_PER_BLOCK 1024
#define BLOCK_SIZE 1024
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;
// static inline double realtime(void) {
//     struct timeval tp;
//     struct timezone tzp;
//     gettimeofday(&tp, &tzp);
//     return tp.tv_sec + tp.tv_usec * 1e-6;
// }
//template<>
//class approx_mul_lut<Eigen::GpuDevice> : public approx_mul_lut_base {
    //public:
        //explicit approx_mul_lut(tensorflow::OpKernelConstruction* context);
        //~approx_mul_lut();
        //auto get_mant_mul_lut_text_() -> cudaTextureObject_t& {
            //return mant_mul_lut_text_;
        //}
        //auto get_exp_mul_lut_text_() -> cudaTextureObject_t& {
            //return exp_mul_lut_text_;
        //}
////   private:
////       uint32_t* mant_mul_lut_cuda_;
////       uint32_t* exp_mul_lut_cuda_;
////       cudaTextureObject_t mant_mul_lut_text_;
////       cudaTextureObject_t exp_mul_lut_text_;
//
//};
//approx_mul_lut<GPUDevice>::approx_mul_lut(OpKernelConstruction * context):
            //approx_mul_lut_base(context){
//
    //gpuErrchk(cudaMalloc(&mant_mul_lut_cuda_, 
            //mant_mul_lut_.size() * sizeof(uint32_t)));
    //std::cout << "size of lut: "<< mant_mul_lut_.size() << std::endl;
    //gpuErrchk(cudaMemcpy(mant_mul_lut_cuda_, mant_mul_lut_.data(),
            //mant_mul_lut_.size()*sizeof(uint32_t), 
            //cudaMemcpyHostToDevice));
    //cudaResourceDesc mant_mul_lut_res_desc;
    //memset(&mant_mul_lut_res_desc, 0, sizeof(cudaResourceDesc));
    //mant_mul_lut_res_desc.resType = cudaResourceTypeLinear;
    //mant_mul_lut_res_desc.res.linear.devPtr = mant_mul_lut_cuda_;
    //mant_mul_lut_res_desc.res.linear.desc.f = 
        //cudaChannelFormatKindUnsigned;
    //mant_mul_lut_res_desc.res.linear.desc.x = 32;
    //mant_mul_lut_res_desc.res.linear.sizeInBytes = 
        //mant_mul_lut_.size() * sizeof(uint32_t);
    //
    //cudaTextureDesc mant_mul_text_desc;
    //memset(&mant_mul_text_desc, 0, sizeof(cudaTextureDesc));
    //mant_mul_text_desc.readMode = cudaReadModeElementType;
        //
    //gpuErrchk(cudaCreateTextureObject(&mant_mul_lut_text_, &mant_mul_lut_res_desc, 
            //&mant_mul_text_desc, nullptr));                
    //std::cout << "mant mul texture created" << std::endl;
//
    //gpuErrchk(cudaMalloc(&exp_mul_lut_cuda_, 
            //exp_mul_lut_.size() * sizeof(uint32_t)));
    //gpuErrchk(cudaMemcpy(exp_mul_lut_cuda_, exp_mul_lut_.data(),
            //exp_mul_lut_.size()*sizeof(uint32_t), 
            //cudaMemcpyHostToDevice));
    //std::cout << "size of exp lut: "<< exp_mul_lut_.size() << std::endl;
    //cudaResourceDesc exp_mul_lut_res_desc;
    //memset(&exp_mul_lut_res_desc, 0, sizeof(cudaResourceDesc));
    //exp_mul_lut_res_desc.resType = cudaResourceTypeLinear;
    //exp_mul_lut_res_desc.res.linear.devPtr = exp_mul_lut_cuda_;
    //exp_mul_lut_res_desc.res.linear.desc.f = 
        //cudaChannelFormatKindUnsigned;
    //exp_mul_lut_res_desc.res.linear.desc.x = 32;
    //exp_mul_lut_res_desc.res.linear.sizeInBytes = 
        //exp_mul_lut_.size() * sizeof(uint32_t);
    //
    //cudaTextureDesc exp_mul_text_desc;
    //memset(&exp_mul_text_desc, 0, sizeof(cudaTextureDesc));
    //exp_mul_text_desc.readMode = cudaReadModeElementType;
        //
    //gpuErrchk(cudaCreateTextureObject(&exp_mul_lut_text_, &exp_mul_lut_res_desc, 
            //&exp_mul_text_desc, nullptr));                
    //std::cout << "exp mul texture created" << std::endl;
//};
//
//approx_mul_lut<GPUDevice>::~approx_mul_lut(){
    //cudaDestroyTextureObject(mant_mul_lut_text_);
    //cudaFree(mant_mul_lut_cuda_);
    //std::cout << "texture freed" << std::endl;
    //cudaDestroyTextureObject(exp_mul_lut_text_);
    //cudaFree(exp_mul_lut_cuda_);
    //std::cout << "texture freed" << std::endl;
//};


__device__ float bitmasking(float num){
	int mask = 0xffff0000;
	int b = *(int*)&num;
    int masked = b&mask;
    float ret  = *(float*)&masked;
	return ret;
}

//=============================================================================
//===============================IM2COL KERNEL=================================
//=============================================================================
template <typename T>
__global__ void im2col(const T* in,int c, int w, int h, int ow, int oh,
        int kw, int kh, int pw, int ph, int sw, int sh, int dw, int dh, int po,
        int pc, T* out
        ) {
    unsigned pl = kw * kh * c;
    for(unsigned tId = blockIdx.x * blockDim.x + threadIdx.x;
            tId < pc*pl; tId += blockDim.x * gridDim.x) {
    unsigned patchId = (tId + po*pl) / pl;
    unsigned outB    = (patchId / ow) / oh;
    unsigned outH    = (patchId / ow) % oh;
    unsigned outW    = patchId % ow;

    unsigned valueId = (tId + po*pl) % pl;
    unsigned offsetH = valueId / (kw * c);
    unsigned offsetW = (valueId / c) % kw;
    unsigned offsetC = valueId % c;

    int inH = outH * sh - ph + offsetH * dh;
    int inW = outW * sw - pw + offsetW * dw;

    if(inH >= 0 && inW >= 0 && inH < h && inW < w)
        out[tId] = in[((outB * h + inH) * w + inW) * c + offsetC];
    else
        out[tId] = T(0);
    }
}
//=============================================================================
//=============================================================================
//=============================================================================
template <typename T>
void im2colLauncher(
    const GPUDevice &d,
    const T* im,
    const int batch,
    const int in_row,
    const int in_col,
    const int out_row,
    const int out_col,
    const int out_depth,
    const int in_depth,
    const int filter_row,
    const int filter_col,
    const int stride_row,
    const int stride_col,
    // Padding
    const int left_offset,
    const int top_offset,
    const int dw,
    const int dh,
    T* data_col)
{

    unsigned pl = filter_row * filter_col * in_depth;
    unsigned blockSize = 256;
    unsigned gridSize  = (batch * pl + blockSize - 1) / blockSize;
    im2col<T><<<gridSize,blockSize,0,d.stream()>>>(im, in_depth, in_col, in_row, out_col, out_row, filter_col, filter_row,  left_offset,top_offset, stride_col, stride_row,dw,dh,0,batch*out_row*out_col,data_col);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}
template <typename T>
__global__ void test(cudaTextureObject_t lut, T *arr, int pos){
    arr[pos] = (T)tex1Dfetch<uint32_t>(lut, pos); 
}
template <typename T>
void ConvamKernelLauncher(
    const GPUDevice &d,
    const T* inputs,
    const T* filter,
    T* im2col,
    const int batch,
    const int in_row,
    const int in_col,
    const int out_row,
    const int out_col,
    const int out_depth,
    const int in_depth,
    const int filter_row,
    const int filter_col,
    const int stride_row,
    const int stride_col,
    // Padding
    const int left_offset,
    const int top_offset,
    const int padding,
    T* output,
    approx_mul_lut<GPUDevice>& mul_lut
  ){

    if (filter_row == 1 && filter_col == 1 && stride_row == 1 &&
        stride_col == 1) {
        // The kernel is 1x1.
        const int m = batch * in_row * in_col;
        const int n = out_depth;
        const int k = in_depth;
        const int lda = k;
        const int ldb = n;
        const int ldc = n;
        //const int size = m*n;
        dim3 blockSize(16, 16, 1);
        dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
        gemm<T><<<gridSize,blockSize,0,d.stream()>>>(m,n,k,inputs,lda,filter,ldb,output, ldc, mul_lut.get_mant_mul_lut_text_(), mul_lut.get_exp_mul_lut_text_(), mul_lut.get_mant_mul_lut_());
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        return;
    } else if (filter_row == in_row && filter_col== in_col &&
               padding == 1) {
         // The input data and filter have the same height/width.
         const int m = batch;
         const int n = out_depth;
         const int k = in_depth*in_col*in_row;
         const int lda = k;
         const int ldb = out_depth;
         const int ldc = out_depth;
         //const int size = m*n;
         dim3 blockSize(16, 16, 1);
         dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
         gemm<T><<<gridSize,blockSize,0,d.stream()>>>(m,n,k,inputs,lda,filter,ldb,output,ldc, mul_lut.get_mant_mul_lut_text_(), mul_lut.get_exp_mul_lut_text_(), mul_lut.get_mant_mul_lut_());
         gpuErrchk( cudaPeekAtLastError() );
         gpuErrchk( cudaDeviceSynchronize() );
         return;
    }
    im2colLauncher<T>(d,inputs, batch, in_row, in_col, out_row, out_col,out_depth, in_depth, filter_row, filter_col, stride_row, stride_col, left_offset,top_offset, 1,1 ,im2col);
    const size_t m = batch*out_row*out_col;
    const size_t n = out_depth;
    const size_t k = filter_col * filter_row * in_depth;
    const size_t lda = k;
    const size_t ldb = out_depth;
    const size_t ldc = out_depth;
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
    gemm<T><<<gridSize,blockSize,0,d.stream()>>>(m,n,k,im2col,lda,filter,ldb,output,ldc, mul_lut.get_mant_mul_lut_text_(), mul_lut.get_exp_mul_lut_text_(), mul_lut.get_mant_mul_lut_());
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}


void loadLUTandCreateTextureObject(uint32_t* mant_lut, uint32_t* exp_lut, 
        cudaTextureObject_t* mant_lut_texture,
        cudaTextureObject_t* exp_lut_texture){
    // load into vector
    // copy to cuda mem
    // create texture obj


    // mant mul
    std::vector<uint32_t> approx_mnt_mul_lookuptable = {};
    std::ifstream file("mbmmantmul16.bin", std::ios::in|std::ios::binary);
    if(file.fail()) {
        std::cerr << "file mbmmantmul16.bin failed" << std::endl;
        exit(1);
    } 
    if(!file.is_open()) { 
        std::cerr << "file mbmmantmul16.bin open failed" << std::endl;
        exit(1);
    }
    approx_mnt_mul_lookuptable.resize(128*128);
    file.read(reinterpret_cast<char *>(approx_mnt_mul_lookuptable.data()), 
            approx_mnt_mul_lookuptable.size() * sizeof(uint32_t));

    cudaMemcpy(mant_lut, approx_mnt_mul_lookuptable.data(), 
            approx_mnt_mul_lookuptable.size() * sizeof(uint32_t),
            cudaMemcpyHostToDevice);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    cudaResourceDesc lookupTableResDesc;
    memset(&lookupTableResDesc, 0, sizeof(cudaResourceDesc));
    lookupTableResDesc.resType = cudaResourceTypeLinear;
    lookupTableResDesc.res.linear.devPtr = mant_lut;
    lookupTableResDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    lookupTableResDesc.res.linear.desc.x = 32;
    lookupTableResDesc.res.linear.sizeInBytes = 
        approx_mnt_mul_lookuptable.size() * sizeof(uint32_t);
    
    cudaTextureDesc lookupTableTexDesc;
    memset(&lookupTableTexDesc, 0, sizeof(cudaTextureDesc));
    lookupTableTexDesc.readMode = cudaReadModeElementType;
    
    gpuErrchk(cudaCreateTextureObject(mant_lut_texture, &lookupTableResDesc,
            &lookupTableTexDesc, nullptr));
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // exp lut
    std::vector<uint32_t> exp_lookuptable = {};
    std::ifstream exp_file("exp.bin", std::ios::in|std::ios::binary);
    if(exp_file.fail()) exit(1);
    if(!exp_file.is_open()) exit(1);
    exp_lookuptable.resize(2*2*256);
    exp_file.read(reinterpret_cast<char *>(exp_lookuptable.data()),
            exp_lookuptable.size() * sizeof(uint32_t));

    cudaMemcpy(exp_lut, exp_lookuptable.data(),
            exp_lookuptable.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    
    cudaResourceDesc exp_lookupTableResDesc;
    memset(&exp_lookupTableResDesc, 0, sizeof(cudaResourceDesc));
    exp_lookupTableResDesc.resType = cudaResourceTypeLinear;
    exp_lookupTableResDesc.res.linear.devPtr = exp_lut;
    exp_lookupTableResDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    exp_lookupTableResDesc.res.linear.desc.x = 32;
    exp_lookupTableResDesc.res.linear.sizeInBytes = 
        exp_lookuptable.size() * sizeof(uint32_t);
    
    cudaTextureDesc exp_lookupTableTexDesc;
    memset(&exp_lookupTableTexDesc, 0, sizeof(cudaTextureDesc));
    exp_lookupTableTexDesc.readMode = cudaReadModeElementType;
    
    cudaCreateTextureObject(exp_lut_texture, &exp_lookupTableResDesc,
            &exp_lookupTableTexDesc, nullptr);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#define loop1D(i,n) for(int i = 0; i < n; ++i)
#define loop1Da(i,n,inc) for(int i = 0; i < n; i+=inc)
// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
void ConvamFunctor<GPUDevice, T>::operator()(const GPUDevice& d,
        const T* input_data, T* output_data, const int batch,
        const int out_rows, const int out_cols, const int out_depth,
        const int stride_cols, const int stride_rows,
        const int filter_left_offset, const int filter_top_offset,
        const int filter_rows, const int filter_cols, const int in_depth,
        const int input_cols, const int input_rows, const T* filter,
        T* im2col, const int padding, approx_mul_lut<GPUDevice>& mul_lut
        ) {
    auto mul_text = mul_lut.get_mant_mul_lut_text_();
    auto exp_text = mul_lut.get_exp_mul_lut_text_();
    // this is a very primitive tiling function. I mean VERY.
    //TODO Simplify the cases
    int const oneinputsize = input_rows * input_cols * in_depth;
    int const oneoutputsize = out_rows* out_cols * out_depth;
    if(filter_cols == 1 && filter_rows == 1 &&
            stride_rows == 1 && stride_cols){
        size_t const block_size = 16;
        size_t const max_batch = (65536*block_size + 1 - block_size)/(input_rows*input_cols);
        if ((size_t)batch <= max_batch) {
            ConvamKernelLauncher<T>(d,
                    input_data,
                    filter,
                    im2col,
                    batch,
                    input_rows,
                    input_cols,
                    out_rows,
                    out_cols,
                    out_depth,
                    in_depth,
                    filter_rows,
                    filter_cols,
                    stride_rows,
                    stride_cols,
                    filter_left_offset,
                    filter_top_offset,
                    padding,
                    output_data,
                    mul_lut
                    );
        } else {
            loop1Da(i, batch, max_batch) {
                size_t const ibatch =  (size_t)(batch - 1 - i) >= max_batch ? max_batch : batch - i;
                ConvamKernelLauncher<T>(d,
                    input_data + i * oneinputsize,
                    filter,
                    im2col,
                    ibatch,
                    input_rows,
                    input_cols,
                    out_rows,
                    out_cols,
                    out_depth,
                    in_depth,
                    filter_rows,
                    filter_cols,
                    stride_rows,
                    stride_cols,
                    filter_left_offset,
                    filter_top_offset,
                    padding,
                    output_data + i * oneoutputsize,
                    mul_lut
                    );
            }
        }
    } else if(filter_rows == input_rows && filter_cols == input_cols&& padding == 1){
            ConvamKernelLauncher<T>(d,
                    input_data,
                    filter,
                    im2col,
                    batch,
                    input_rows,
                    input_cols,
                    out_rows,
                    out_cols,
                    out_depth,
                    in_depth,
                    filter_rows,
                    filter_cols,
                    stride_rows,
                    stride_cols,
                    filter_left_offset,
                    filter_top_offset,
                    padding,
                    output_data,
                    mul_lut
                    );
    } else {
        size_t const block_size = 16;
        size_t const max_batch = (65536*block_size + 1 - block_size)/(out_rows*out_cols);
        if ((size_t)batch <= max_batch) {
            ConvamKernelLauncher<T>(d,
                    input_data,
                    filter,
                    im2col,
                    batch,
                    input_rows,
                    input_cols,
                    out_rows,
                    out_cols,
                    out_depth,
                    in_depth,
                    filter_rows,
                    filter_cols,
                    stride_rows,
                    stride_cols,
                    filter_left_offset,
                    filter_top_offset,
                    padding,
                    output_data,
                    mul_lut
                    );
        } else {
            loop1Da(i, batch, max_batch){
                size_t const ibatch =  (size_t)(batch - 1 - i) >= max_batch ? max_batch : batch - i;
                ConvamKernelLauncher<T>(d,
                    input_data + i * oneinputsize,
                    filter,
                    im2col,
                    ibatch,
                    input_rows,
                    input_cols,
                    out_rows,
                    out_cols,
                    out_depth,
                    in_depth,
                    filter_rows,
                    filter_cols,
                    stride_rows,
                    stride_cols,
                    filter_left_offset,
                    filter_top_offset,
                    padding,
                    output_data + i * oneoutputsize,
                    mul_lut
                    );
            }
        }
    }

}




template <typename T>
__global__ void im2col_filtergrad(const T *in, int batch,
    int c, int w, int h, int ow, int oh,
    int kw, int kh, int pw, int ph, int sw, int sh,
    int dw, int dh, int po, int pc, T *out){
    unsigned pl = batch * oh * ow;
    for(unsigned tId = blockIdx.x * blockDim.x + threadIdx.x;
            tId < pc*pl; tId += blockDim.x * gridDim.x){
        unsigned patchId = (tId + po*pl) / pl;
        unsigned outB    = (patchId / c) / kw; // kh
        unsigned outH    = (patchId / c) % kw; // kw
        unsigned outW    = patchId % c; // c

        unsigned valueId = (tId + po*pl) % pl; // element position in window
        unsigned offsetH = valueId / (ow * oh);//ob
        unsigned offsetW = (valueId / ow) % oh;//oh
        unsigned offsetC = valueId % ow; //ow

        int inH = outB * 1 - ph + offsetW * sh;
        int inW = outH * 1 - pw + offsetC * sw;
        if(inH >= 0 && inW >= 0 && inH < h && inW < w)
            out[tId] = in[((offsetH * h + inH) * w + inW) * c + outW];
        else
            out[tId] = T(0);

    }

}
template <typename T>
void im2colLauncher_filtergrad(
    const GPUDevice &d,
    const T* im,
    const int batch,
    const int in_row,
    const int in_col,
    const int out_row,
    const int out_col,
    const int out_depth,
    const int in_depth,
    const int filter_row,
    const int filter_col,
    const int stride_row,
    const int stride_col,
    // Padding
    const int left_offset,
    const int top_offset,
    const int dw,
    const int dh,
    T* data_col)
{

    unsigned pl = batch * out_row * out_col;
    unsigned blockSize = 256;
    unsigned gridSize  = (filter_row * pl + blockSize - 1) / blockSize;

    im2col_filtergrad<T><<<gridSize,blockSize,0,d.stream()>>>(im, batch, in_depth, in_col, in_row, out_col, out_row, filter_col, filter_row,  left_offset,top_offset, stride_col, stride_row,dw,dh,0,filter_row*filter_col*in_depth,data_col);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}


template <typename T>
void ConvamFilterGradKernelLauncher(
    const GPUDevice &d,
    const T* input,
    const T* grad,
    T* im2col,
    const int input_height,
    const int input_width,
    const int batch,
    const int in_depth,
    const int grad_width,
    const int grad_height,
    const int grad_channel,
    const int filter_left_offset,
    const int filter_top_offset,
    const int stride_row,
    const int stride_col,
    const int filter_width,
    const int filter_height,
    T* out, 
    approx_mul_lut<GPUDevice>& mul_lut
){

    im2colLauncher_filtergrad<T>(d,input,batch,input_height,input_width,grad_height,grad_width,grad_channel,in_depth,filter_height,filter_width,stride_row,stride_col,\
    filter_left_offset,filter_top_offset,1,1,im2col);
    const size_t m = filter_height*filter_width*in_depth;
    const size_t n = grad_channel;
    const size_t k = batch*grad_height*grad_width;
    const size_t lda = k;
    const size_t ldb = n;
    const size_t ldc = n;
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
    gemm<T><<<gridSize,blockSize,0,d.stream()>>>(m,n,k,im2col,lda,grad,ldb,out,ldc,mul_lut.get_mant_mul_lut_text_(), mul_lut.get_exp_mul_lut_text_(), mul_lut.get_mant_mul_lut_());
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
};

template <typename T>
void ConvamFilterGradFunctor<Eigen::GpuDevice, T>::operator()(
        const Eigen::GpuDevice& d, const T* input, const T* grad,
        T* im2col, const int input_rows, const int input_cols,
        const int batch, const int in_depth, const int out_cols,
        const int out_rows,const int out_depth, const int filter_left_offset,
        const int filter_top_offset, const int stride_rows,
        const int stride_cols, const int filter_cols, const int filter_rows,
        T* output, approx_mul_lut<GPUDevice>& mul_lut
        ) {
    ConvamFilterGradKernelLauncher<T>(
            d,
            input,
            grad,
            im2col,
            input_rows,
            input_cols,
            batch,
            in_depth,
            out_cols,
            out_rows,
            out_depth,
            filter_left_offset,
            filter_top_offset,
            stride_rows,
            stride_cols,
            filter_cols,
            filter_rows,
            output,
            mul_lut
            );
}
template <typename T>
__global__ void im2col_inputgrad(const T *in, int c, int w, int h, int ow,
        int oh, int kw, int kh, int pw, int ph, int sw, int sh, int dw, int dh,
        int po, int pc, T *out)
{
	unsigned pl = kw * kh * c;
	for(unsigned tId = blockIdx.x * blockDim.x + threadIdx.x;
            tId < pc*pl; tId += blockDim.x * gridDim.x)
	{
		unsigned patchId = (tId + po*pl) / pl;
		unsigned outB    = (patchId / ow) / oh;
		unsigned outH    = (patchId / ow) % oh;
		unsigned outW    = patchId % ow;

		unsigned valueId = (tId + po*pl) % pl;
		unsigned offsetH = valueId / (kw * c);
		unsigned offsetW = (valueId / c) % kw;
		unsigned offsetC = valueId % c;

		int inH = outH * 1 - ph + offsetH * dh;
		int inW = outW * 1 - pw + offsetW * dw;

		if(inH >= 0 && inW >= 0 && inH < h && inW < w) {
			if (inH%sh == 0 && inW%sw == 0)
			{
                unsigned orign_h = (h-1)/sh+1;
                unsigned orign_w = (w-1)/sw+1;
                unsigned orign_h_idx = inH/sh;
                unsigned orign_w_idx = inW/sw;
				out[tId] = in[((outB * (orign_h) + orign_h_idx) *
                        (orign_w) + orign_w_idx) * c + offsetC];
			} else
			{
				out[tId] = T(0);
			}
		}
		else {
			out[tId] = T(0);
		}
	}
}
template <typename T>
void im2colLauncher_inputgrad(
    const GPUDevice &d,
    const T* im,
    const int batch,
    const int in_row,
    const int in_col,
    const int out_row,
    const int out_col,
    const int out_depth,
    const int in_depth,
    const int filter_row,
    const int filter_col,
    const int stride_row,
    const int stride_col,
    // Padding
    const int left_offset,
    const int top_offset,
    const int dw,
    const int dh,
    T* data_col)
{

    unsigned pl = filter_row * filter_col * in_depth;
    unsigned blockSize = 256;
    unsigned gridSize  = (batch * pl + blockSize - 1) / blockSize;
    im2col_inputgrad<T><<<gridSize,blockSize,0,d.stream()>>>(im, in_depth, in_col, in_row, out_col, out_row, filter_col, filter_row,  left_offset,top_offset, stride_col, stride_row,dw,dh,0,batch*out_row*out_col,data_col);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}

template <typename T>
void ConvamInputGradKernelLauncher(
    const GPUDevice &d,
    const T* grad,
    T* im2col,
    const int hole_grad_width,
    const int hole_grad_height,
    const int back_pad_top,
    const int back_pad_left,
    const T* filter,
    //reverse and swap dimension 2 and 3 of the filters.s
    T* rsfilter,
    const int filter_height,
    const int filter_width,
    const int output_channel,
    const int stride_rows,
    const int stride_cols,
    // input related
    const int input_batch,
    const int input_height,
    const int input_width,
    const int input_channel,
    T* output,
    approx_mul_lut<GPUDevice>& mul_lut
){

    im2colLauncher_inputgrad<T>(
        d,grad, input_batch, hole_grad_height, hole_grad_width, input_height, input_width,input_channel,output_channel,filter_height,
        filter_width,stride_rows,stride_cols,back_pad_left,back_pad_top,1,1,im2col);
    const size_t m = input_batch*input_height*input_width; //4
    const size_t n = input_channel; //  1
    const size_t k = filter_width * filter_height * output_channel; //4
    const size_t lda = k; //4
    const size_t ldb = input_channel;
    const size_t ldc = input_channel;
    //const int size = m*n;
    dim3 block_size(32,32);
    dim3 grid_size(ceil(filter_width * filter_height/(float)32.0), ceil(output_channel/(float)32.0));
    reverseNswapdim23<T><<<grid_size,block_size>>>(filter_height, filter_width, input_channel, output_channel, rsfilter, filter);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
    gemm<T><<<gridSize,blockSize,0,d.stream()>>>(m,n,k,im2col,lda,rsfilter,ldb,output,ldc,mul_lut.get_mant_mul_lut_text_(), mul_lut.get_exp_mul_lut_text_(), mul_lut.get_mant_mul_lut_());
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

};

template <typename T>
void ConvamInputGradFunctor<Eigen::GpuDevice, T>::operator()(
        const Eigen::GpuDevice& d, const T* grad, T* im2col,
        const int hole_grad_width, const int hole_grad_height,
        const int pad_top, const int pad_left, const T* filter, T* rsfilter,
        const int filter_rows, const int filter_cols, const int out_depth,
        const int stride_rows, const int stride_cols, const int batch,
        const int input_rows, const int input_cols, const int in_depth,
        T* output, const int out_rows, const int out_cols, 
        approx_mul_lut<GPUDevice>& mul_lut
        ){
    // a very primitive tiling, I mean VERY
    //auto const oneinputsize = input_rows*input_cols*in_depth;
    auto const oneoutputsize = out_rows*out_cols*out_depth;
    size_t const block_size = 16;
    size_t const max_batch = (65536*block_size + 1 - block_size)/
        (input_rows*input_cols);
    if ((size_t)batch <= max_batch) {

        ConvamInputGradKernelLauncher<T>(
                d,
                grad,
                im2col,
                hole_grad_width,
                hole_grad_height,
                pad_top,
                pad_left,
                filter,
                rsfilter,
                filter_rows,
                filter_cols,
                out_depth,
                stride_rows,
                stride_cols,
                batch,
                input_rows,
                input_cols,
                in_depth,
                output,
                mul_lut
                );
    } else {
        loop1Da(i, batch, max_batch){
            size_t const ibatch =  (size_t)(batch - 1 - i) >= max_batch ? max_batch : batch - i;
            ConvamInputGradKernelLauncher<T>(
                     d,
                     grad + i*oneoutputsize,
                     im2col,
                     hole_grad_width,
                     hole_grad_height,
                     pad_top,
                     pad_left,
                     filter,
                     rsfilter,
                     filter_rows,
                     filter_cols,
                     out_depth,
                     stride_rows,
                     stride_cols,
                     ibatch,
                     input_rows,
                     input_cols,
                     in_depth,
                     output + i*oneoutputsize,
                     mul_lut
                );
        }
    }
}
template struct ConvamInputGradFunctor<GPUDevice, float>;
template struct ConvamInputGradFunctor<GPUDevice, int32>;
template struct ConvamFilterGradFunctor<GPUDevice, float>;
template struct ConvamFilterGradFunctor<GPUDevice, int32>;
template struct ConvamFunctor<GPUDevice, float>;
template struct ConvamFunctor<GPUDevice, int32>;
