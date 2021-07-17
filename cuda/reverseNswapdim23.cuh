#ifndef REVERSENSWAPDIM23
#define REVERSENSWAPDIM23

__global__ void reverseNswapdim23(size_t height, size_t width, size_t n_channels, size_t n_filters, float* dest, const float* src );

#endif
