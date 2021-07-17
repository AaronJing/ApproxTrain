#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "gemm.cuh"


#define TILE_DIM 16
__global__ void gemm(size_t m, size_t n, size_t k,
    const float *a, size_t lda, const float *b, size_t ldb,
    float *c, size_t ldc)
{
    float value(0);

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int i = 0; i < (TILE_DIM + k - 1)/TILE_DIM; ++i) {

        if (i*TILE_DIM + threadIdx.x < k && Row < m){
            As[threadIdx.y][threadIdx.x] = a[Row*lda + i*TILE_DIM + threadIdx.x];
        }
        else{
            As[threadIdx.y][threadIdx.x] = float(0);
        }

        if (i*TILE_DIM + threadIdx.y < k && Col < n){
            Bs[threadIdx.y][threadIdx.x] = b[(i*TILE_DIM + threadIdx.y)*ldb + Col];
        }
        else{
            Bs[threadIdx.y][threadIdx.x] = float(0);
        }

        __syncthreads();

        for (int n = 0; n < TILE_DIM; ++n){
            //value += bitmasking(bitmasking(As[threadIdx.y][n])*bitmasking(Bs[n][threadIdx.x]));
            value += MULTIPLY(As[threadIdx.y][n],Bs[n][threadIdx.x]);
            //value += FPMult_SinglePrecision_Rnone_Mitchell(As[threadIdx.y][n],Bs[n][threadIdx.x],MT);
            //value += As[threadIdx.y][n]*Bs[n][threadIdx.x];
        }

        __syncthreads();
    }

    if (Row < m && Col < n) {
        c[((blockIdx.y * blockDim.y  + threadIdx.y)*ldc) + (blockIdx.x * blockDim.x) + threadIdx.x] = value;
    }
}
