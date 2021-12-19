#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "tensorflow/core/framework/types.h"
using namespace tensorflow;

template <typename T>
__global__ void reverseNswapdim23(size_t height, size_t width, size_t n_channels, size_t n_filters, T* dest, const T* src ){
    size_t pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t filter_index = blockIdx.y * blockDim.y + threadIdx.y;
    size_t num_pixels = height*width;
    if(filter_index < n_filters && pixel_index < num_pixels){
        for(size_t channel_idx = 0; channel_idx<n_channels; channel_idx++){
            dest[pixel_index*n_channels*n_filters + filter_index*n_channels + channel_idx] = src[(num_pixels - pixel_index -1)*n_channels*n_filters + channel_idx*n_filters + filter_index];
        }
    }
}

template __global__ void reverseNswapdim23<float>(size_t height, size_t width, size_t n_channels, size_t n_filters, float* dest, const float* src );
template __global__ void reverseNswapdim23<int32>(size_t height, size_t width, size_t n_channels, size_t n_filters, int32* dest, const int32* src );
