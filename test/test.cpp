
#include "../Convam.h"
using GPUDevice = Eigen::GpuDevice;
using CPUDevice = Eigen::ThreadPoolDevice;


// template <typename T>
// void ConvamFilterGradFunctor<Eigen::GpuDevice, T>::operator()(
//         const Eigen::GpuDevice& d, const T* input, const T* grad, 
//         T* im2col, const int input_rows, const int input_cols, 
//         const int batch, const int in_depth, const int out_cols, 
//         const int out_rows,const int out_depth, const int filter_left_offset, 
//         const int filter_top_offset, const int stride_rows, 
//         const int stride_cols, const int filter_cols, const int filter_rows, 
//         T* output
//         ) {

//            printf("FUck"); 
//     // ConvamFilterGradKernelLauncher<T>(
//     //         d,
//     //         input,
//     //         grad,
//     //         im2col,
//     //         input_rows,
//     //         input_cols,
//     //         batch,
//     //         in_depth,
//     //         out_cols,
//     //         out_rows,
//     //         out_depth,
//     //         filter_left_offset,
//     //         filter_top_offset,
//     //         stride_rows,
//     //         stride_cols,
//     //         filter_cols,
//     //         filter_rows,
//     //         output
//     //         );
// }

int main(){
// void ConvamFilterGradFunctor<Eigen::GpuDevice, T>::operator()(
//         const Eigen::GpuDevice& d, const T* input, const T* grad, 
//         T* im2col, const int input_rows, const int input_cols, 
//         const int batch, const int in_depth, const int out_cols, 
//         const int out_rows,const int out_depth, const int filter_left_offset, 
//         const int filter_top_offset, const int stride_rows, 
//         const int stride_cols, const int filter_cols, const int filter_rows, 
//         T* output
//         )

        const int d=0; const float* input; const float* grad;
        float* im2col; const int input_rows=0; const int input_cols=0; 
        const int batch=0; const int in_depth=0; const int out_cols=0; 
        const int out_rows=0;const int out_depth=0; const int filter_left_offset=0; 
        const int filter_top_offset=0; const int stride_rows=0; 
        const int stride_cols=0; const int filter_cols=0; const int filter_rows=0; 
        float* output;

        struct ConvamFilterGradFunctor<GPUDevice, float> op; 

    op.operator()((const Eigen::GpuDevice&)d, input, grad, 
        im2col, input_rows, input_cols, 
        batch, in_depth, out_cols, 
        out_rows,out_depth,  filter_left_offset, 
        filter_top_offset, stride_rows, 
        stride_cols,  filter_cols, filter_rows, 
        output);
    return 0;
}