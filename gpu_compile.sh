#!/bin/bash

TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
echo $TF_CFLAGS
echo $TF_LFLAGS
rm -f *.o *.so

#MULTIPLIER="-DFMBM32_MULTIPLIER=1"
# MULTIPLIER="-DFMBM16_MULTIPLIER=1"
#PROFILE="-DPROFILE=1"
NVCC=nvcc
$NVCC -std=c++11 -O2 -c -o  cuda_kernel.cu.o cuda/cuda_kernel.cu \
  ${TF_CFLAGS[@]} -DGOOGLE_CUDA=1 -DNDEBUG $PROFILE -x cu -Xcompiler -fPIC  --expt-relaxed-constexpr
$NVCC -std=c++11 -O2 -c -o gemm.cu.o cuda/gemm.cu \
  ${TF_CFLAGS[@]} -DGOOGLE_CUDA=1 $MULTIPLIER  -x cu -Xcompiler -fPIC
$NVCC -std=c++11 -O2 -c -o reverseNswapdim23.cu.o cuda/reverseNswapdim23.cu \
  ${TF_CFLAGS[@]} -DGOOGLE_CUDA=1 $MULTIPLIER  -x cu -Xcompiler -fPIC 
g++ -std=c++11 -O2 -shared -o convam_gpu.so Convam.cc   cuda_kernel.cu.o gemm.cu.o reverseNswapdim23.cu.o $PROFILE ${TF_CFLAGS[@]}  -fPIC -L/usr/local/cuda/lib64 -lcudart ${TF_LFLAGS[@]}
rm -f *.o
