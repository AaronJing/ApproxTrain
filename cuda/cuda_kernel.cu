#define EIGEN_USE_GPU
#include <cuda.h>
#include <stdio.h>
#include <fstream>
#include <assert.h>
#include <cuda_fp16.h>
#include <iostream>
#include <chrono>
#include <sys/time.h>
#include "error.cuh"
#include "gemm.cuh"
#include "reverseNswapdim23.cuh"
using namespace std;
#define THREADS_PER_BLOCK 1024
#define BLOCK_SIZE 1024

static inline double realtime(void) {
    struct timeval tp;
    struct timezone tzp;
    gettimeofday(&tp, &tzp);
    return tp.tv_sec + tp.tv_usec * 1e-6;
}

__device__ float halfmul(const float a, const float b){
      half A = __float2half(a);
      half B = __float2half(b);
      half C;
  #if __CUDA_ARCH__ >= 530
      C = __hmul(A, B);
  #else
      C = __float2half(__half2float(A)*__half2float(B));
  #endif
      float c = __half2float(C);

    return c;
}

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
/*po patch offset, pc patch count*/
__global__ void im2col_improved(const float *in,
    int c, int w, int h, int ow, int oh,
    int kw, int kh, int pw, int ph, int sw, int sh,
    int dw, int dh, int po, int pc, float *out)
{
//pc = ow * oh * batch aka m dimension
unsigned pl = kw * kh * c;
for(unsigned tId = blockIdx.x * blockDim.x + threadIdx.x; tId < pc*pl; tId += blockDim.x * gridDim.x)
{
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
        out[tId] = float(0);

}

}
//=============================================================================
//=============================================================================
//=============================================================================
void im2colLauncher_Improved(
    const float* im,
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
    float* data_col)
{

    unsigned pl = filter_row * filter_col * in_depth;
    unsigned blockSize = 256;
    unsigned gridSize  = (batch * pl + blockSize - 1) / blockSize;
    im2col_improved<<<gridSize,blockSize,0>>>(im, in_depth, in_col, in_row, out_col, out_row, filter_col, filter_row,  left_offset,top_offset, stride_col, stride_row,dw,dh,0,batch*out_row*out_col,data_col);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}

// //=============================================================================
// //===============================IM2COL KERNEL=================================

__global__ void im2col_improved_filtergrad(const float *in, int batch,
    int c, int w, int h, int ow, int oh,
    int kw, int kh, int pw, int ph, int sw, int sh,
    int dw, int dh, int po, int pc, float *out)
{
//unsigned pc = kernel_height * kernel_width * channel_in

unsigned pl = batch * oh * ow;
for(unsigned tId = blockIdx.x * blockDim.x + threadIdx.x; tId < pc*pl; tId += blockDim.x * gridDim.x)
{
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
        out[tId] = float(0);

}

}
//=============================================================================
//=============================================================================
//=============================================================================
void im2colLauncher_Improved_filtergrad(
    const float* im,
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
    float* data_col)
{

    unsigned pl = batch * out_row * out_col;
    unsigned blockSize = 256;
    unsigned gridSize  = (filter_row * pl + blockSize - 1) / blockSize;

    im2col_improved_filtergrad<<<gridSize,blockSize,0>>>(im, batch, in_depth, in_col, in_row, out_col, out_row, filter_col, filter_row,  left_offset,top_offset, stride_col, stride_row,dw,dh,0,filter_row*filter_col*in_depth,data_col);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}




void gemm_reference( size_t m, size_t n,
    size_t k, const float* a, size_t lda, const float* b, size_t ldb,
    float* c, size_t ldc){
    const size_t a_i_stride = lda; 
    const size_t b_l_stride = ldb; 
    const size_t c_i_stride = ldc; 
    size_t i, j, l;
    // loop output depth
    for (j = 0; j < n; j++) {
      //loop patch
      for (i = 0; i < m; i++) {
        float total(0);
        //loop filter_value_count
        for (l = 0; l < k; l++) {
          const size_t a_index = ((i * a_i_stride) + l );
          const float a_value = a[a_index];
          // filte
          const size_t b_index = (j  + (l * b_l_stride));
          const float b_value = b[b_index];
          total += (a_value * b_value);
        }
        const size_t c_index = ((i * c_i_stride) + j );
        c[c_index] = total;
      }
    }
  }


void ConvamKernellLauncher(
    const float* inputs,
    const float* filter,
    float* im2col,
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
    float* output
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
      const int size = m*n;
      dim3 blockSize(16, 16, 1);
      dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
      gemm<<<gridSize,blockSize,0>>>(m,n,k,inputs,lda,filter,ldb,output,ldc);
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
      const int size = m*n;
      dim3 blockSize(16, 16, 1);
      dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
      gemm<<<gridSize,blockSize,0>>>(m,n,k,inputs,lda,filter,ldb,output,ldc);
      gpuErrchk( cudaPeekAtLastError() );
gpuErrchk( cudaDeviceSynchronize() );
      return;
    }
    double begin = realtime();
   im2colLauncher_Improved(inputs, batch, in_row, in_col, out_row, out_col,out_depth, in_depth, filter_row, filter_col, stride_row, stride_col, left_offset,top_offset, 1,1 ,im2col);
   cudaDeviceSynchronize();
   double end = realtime();
#ifdef PROFILE
cout << "Forward Im2col time difference = " << end - begin << endl;
#endif
   const size_t m = batch*out_row*out_col; 
   const size_t n = out_depth; 
   const size_t k = filter_col * filter_row * in_depth; 
   const size_t lda = k; 
   const size_t ldb = out_depth;
   const size_t ldc = out_depth;
   dim3 blockSize(16, 16, 1);
   dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
    begin =realtime();
   gemm<<<gridSize,blockSize,0>>>(m,n,k,im2col,lda,filter,ldb,output,ldc);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    end = realtime();
#ifdef PROFILE
    cout << "Forward gemm time difference = " << end - begin << endl;
#endif


  }

void Im2col(const float* input_data, const int depth, const int height,
    const int width, const int filter_h, const int filter_w,
    const int pad_t, const int pad_l, const int pad_b, const int pad_r,
    const int stride_h, const int stride_w, float* col_data) {
    int height_col = 3;
    int width_col = 3;

    int h_pad = -pad_t;
    for (int h = 0; h < height_col; ++h) {
        int w_pad = -pad_l;
            for (int w = 0; w < width_col; ++w) {
                for (int ih = h_pad; ih < h_pad + filter_h; ++ih) {
                    for (int iw = w_pad; iw < w_pad + filter_w; ++iw) {
                    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
                        memcpy(col_data, input_data + (ih * width + iw) * depth,
                            sizeof(float) * depth);
                    } else {
                        // This should be simply padded with zero.
                        memset(col_data, 0, sizeof(float) * depth);
                    }
                col_data += depth;
                }
            }
        w_pad += stride_w;
        }
        h_pad += stride_h;
    }
}



void ConvamFilterGradKernelLauncher(
    const float* input,
    const float* grad,
    float* im2col,
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
    float* out
){

    double begin = realtime();
    im2colLauncher_Improved_filtergrad(input,batch,input_height,input_width,grad_height,grad_width,grad_channel,in_depth,filter_height,filter_width,stride_row,stride_col,\
    filter_left_offset,filter_top_offset,1,1,im2col);
    double end = realtime();
#ifdef PROFILE
    cout << "Filter gradient im2col difference = " << end - begin << endl;
#endif
    const size_t m = filter_height*filter_width*in_depth; 
    const size_t n = grad_channel; 
    const size_t k = batch*grad_height*grad_width; 
    const size_t lda = k; 
    const size_t ldb = n;
    const size_t ldc = n;

    begin =realtime();
    dim3 blockSize(16, 16, 1);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);
    gemm<<<gridSize,blockSize,0>>>(m,n,k,im2col,lda,grad,ldb,out,ldc);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    end = realtime();

#ifdef PROFILE
    cout << "Filter gradient gemm difference = " << end - begin << endl;
#endif


};



__global__ void im2col_improved_inputgrad(const float *in,
    int c, int w, int h, int ow, int oh,
    int kw, int kh, int pw, int ph, int sw, int sh,
    int dw, int dh, int po, int pc, float *out)
{
	//pc = ow * oh * batch aka m dimension
	unsigned pl = kw * kh * c;
	for(unsigned tId = blockIdx.x * blockDim.x + threadIdx.x; tId < pc*pl; tId += blockDim.x * gridDim.x)
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
				out[tId] = in[((outB * (h/sh) + inH/sh) * (w/sw) + inW/sw) * c + offsetC];
				// printf("tID %d %d %d %d %d %d %d %d %d %d\n",outB , h ,inH,sh ,w ,inW,sw ,c ,offsetC,((outB * h + inH/sh) * w + inW/sw) * c + offsetC);
			} else
			{
				out[tId] = float(0);
			}
		}
		else { 
			out[tId] = float(0); 
		}
			

	}

}
void im2colLauncher_Improved_inputgrad(
    const float* im,
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
    float* data_col)
{

    unsigned pl = filter_row * filter_col * in_depth;
    unsigned blockSize = 256;
    unsigned gridSize  = (batch * pl + blockSize - 1) / blockSize;
    im2col_improved_inputgrad<<<gridSize,blockSize,0>>>(im, in_depth, in_col, in_row, out_col, out_row, filter_col, filter_row,  left_offset,top_offset, stride_col, stride_row,dw,dh,0,batch*out_row*out_col,data_col);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

}


void ConvamInputGradKernelLauncher(
    // grad needs pading and holes
    // im2col input
    const float* grad,
    float* holed_grad,
    float* im2col,
    const int real_grad_height,
    const int real_grad_width,
    const int hole_grad_width,
    const int hole_grad_height,
    const int back_pad_top,
    const int back_pad_left,
    const int back_pad_bottom,
    const int back_pad_right,
    const float* filter,
    //reverse and swap dimension 2 and 3 of the filters.s
    float* rsfilter,
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
    float* output
){

    double begin1 = realtime();
//     if(hole_grad_height!=real_grad_height||hole_grad_width!=real_grad_width){
//         // float holed[input_batch*hole_grad_width*hole_grad_height*output_channel];
// 		// std::cout << "ok" << std::endl;
//         const int holed_size = hole_grad_width*hole_grad_height*output_channel;
//         const int real_size = output_channel*real_grad_height*real_grad_width;
//         dim3 dim_grid(ceil((float)holed_size/BLOCK_SIZE),input_batch);
//         inserthole<<<dim_grid,BLOCK_SIZE>>>( grad, real_grad_height, real_grad_width, output_channel, hole_grad_height, hole_grad_width, holed_size,
//             stride_rows, stride_cols, real_size, holed_grad);

//             gpuErrchk( cudaPeekAtLastError() );
//             gpuErrchk( cudaDeviceSynchronize() );
//        im2colLauncher_Improved(
//         holed_grad, input_batch, hole_grad_height, hole_grad_width, input_height, input_width,input_channel,output_channel,filter_height,
//         filter_width,1,1,back_pad_left,back_pad_top,1,1,im2col);
//    } else {
    im2colLauncher_Improved_inputgrad(
        grad, input_batch, hole_grad_height, hole_grad_width, input_height, input_width,input_channel,output_channel,filter_height,
        filter_width,stride_rows,stride_cols,back_pad_left,back_pad_top,1,1,im2col);
    // }
    double end1 = realtime();
	// float temp1[output_channel*filter_width*filter_height*input_height*input_width*input_batch];
	// cudaMemcpy(temp1,grad,sizeof(float)*output_channel*real_grad_height*real_grad_width*input_batch,cudaMemcpyDeviceToHost);
	// for (size_t i = 0; i < output_channel*real_grad_height*real_grad_width*input_batch; i++)
	// {
	// 	printf("%f\n",temp1[i]);
	// }
	// float temp[output_channel*filter_width*filter_height*input_height*input_width*input_batch];
	// cudaMemcpy(temp,im2col,sizeof(float)*output_channel*filter_width*filter_height*input_height*input_width*input_batch,cudaMemcpyDeviceToHost);
	// for (size_t i = 0; i < output_channel*filter_width*filter_height*input_height*input_width*input_batch; i++)
	// {
	// 	printf("%f\n",temp[i]);
	// }
	
#ifdef PROFILE
    cout << "Error backpropagation: Im2Col time difference = " << end1 - begin1 << endl;
#endif


    const size_t m = input_batch*input_height*input_width; //4
    const size_t n = input_channel; //  1
    const size_t k = filter_width * filter_height * output_channel; //4
    const size_t lda = k; //4
    const size_t ldb = input_channel;
    const size_t ldc = input_channel;
    const int size = m*n;
    double begin =realtime();
    dim3 block_size(32,32);
    dim3 grid_size(ceil(filter_width * filter_height/(float)32.0), ceil(output_channel/(float)32.0));
    reverseNswapdim23<<<grid_size,block_size>>>(filter_height, filter_width, input_channel, output_channel, rsfilter, filter);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    dim3 blockSize(16, 16, 1);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y, 1);

    gemm<<<gridSize,blockSize,0>>>(m,n,k,im2col,lda,rsfilter,ldb,output,ldc);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    double end = realtime();

#ifdef PROFILE
    cout << "Error backpropagation: Gemm inverse time = " << end - begin << endl;
#endif

};
