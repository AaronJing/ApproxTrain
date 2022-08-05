CC       = gcc
CXX      = g++
CPPFLAGS += -DGOOGLE_CUDA=1 -I.
CFLAGS   += -g -Wall -O2 -std=c++11  -fPIC
LDFLAGS  +=

TF_CFLAGS := $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CPPFLAGS += $(TF_CFLAGS)
LDFLAGS += $(TF_LFLAGS)

CONV_BINARY = convam_gpu.so
CONV_OBJ = convam.o
DENSE_BINARY = denseam_gpu.so
DENSE_OBJ = denseam.o

CUDA_ROOT = /usr/local/cuda
CUDA_LIB ?= $(CUDA_ROOT)/lib64
CONV_CUDA_OBJ = cuda_kernel.cu.o  gemm.cu.o reverseNswapdim23.cu.o approx_mul_lut.cu.o 
NVCC ?= nvcc
CUDA_CFLAGS += -g  -O2 -std=c++11 $(CUDA_ARCH) -Xcompiler -Wall -Xcompiler -fPIC  -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr
CUDA_LDFLAGS = -L$(CUDA_LIB) -lcudart
CONV_OBJ += $(CONV_CUDA_OBJ)
DENSE_CUDA_OBJ = denseam_kernel.cu.o approx_mul_lut.cu.o
DENSE_OBJ += $(DENSE_CUDA_OBJ)
ifeq  ($(MULTIPLIER),)
    MULTIPLIER_CPPFLAG =
else
	MULTIPLIER_CPPFLAG = -D $(MULTIPLIER)=1
endif

.PHONY: clean test

all: $(CONV_BINARY) $(DENSE_BINARY)

convam: $(CONV_BINARY)
	
denseam: $(DENSE_BINARY)

$(CONV_BINARY): $(CONV_OBJ)
	$(CXX) $(CFLAGS) -shared $(CONV_OBJ) $(LDFLAGS) $(CUDA_LDFLAGS) -o $@

$(DENSE_BINARY): $(DENSE_OBJ)
	$(CXX) $(CFLAGS) -shared $(DENSE_OBJ) $(LDFLAGS) $(CUDA_LDFLAGS) -o $@

test_bin: $(CONV_OBJ)
	$(CXX)  $(CFLAGS) $(CPPFLAGS) $(CONV_OBJ) test/test.cpp $(LDFLAGS) $(CUDA_LDFLAGS) -o $@

convam.o: convam.cc convam.h
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(MULTIPLIER_CPPFLAG) $< -c -o $@
denseam.o: denseam.cc denseam.h 
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(MULTIPLIER_CPPFLAG) $< -c -o $@

# header deps
mul_inc_deps = cuda/AMsimulator.inl 

# cuda stuff
denseam_kernel.cu.o : cuda/denseam_kernel.cu cuda/error.cuh $(mul_inc_deps) 
	$(NVCC) -x cu $(CUDA_CFLAGS) $(CPPFLAGS) $(MULTIPLIER_CPPFLAG) -c $< -o $@
cuda_kernel.cu.o: cuda/cuda_kernel.cu cuda/gpu_kernel_helper.h cuda/error.cuh cuda/gemm.cuh cuda/reverseNswapdim23.cuh 
	$(NVCC) -x cu $(CUDA_CFLAGS) $(CPPFLAGS) --expt-relaxed-constexpr -c $< -o $@

gemm.cu.o: cuda/gemm.cu $(mul_inc_deps)
	$(NVCC) -x cu $(CUDA_CFLAGS) $(CPPFLAGS) $(MULTIPLIER_CPPFLAG) -c $< -o $@

reverseNswapdim23.cu.o: cuda/reverseNswapdim23.cu
	$(NVCC) -x cu $(CUDA_CFLAGS) $(CPPFLAGS) -c $< -o $@

approx_mul_lut.cu.o: cuda/approx_mul_lut.cu cuda/error.cuh 
	$(NVCC) -x cu $(CUDA_CFLAGS) $(CPPFLAGS) -c $< -o $@

clean:
	rm -f *.o *.so

test: $(CONV_BINARY) $(DENSE_BINARY)
	python3 mnist_example.py
