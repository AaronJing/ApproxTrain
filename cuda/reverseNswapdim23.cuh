#ifndef REVERSENSWAPDIM23
#define REVERSENSWAPDIM23
template <typename T>
__global__ void reverseNswapdim23(size_t height, size_t width, size_t n_channels, size_t n_filters, T* dest, const T* src );

#endif
