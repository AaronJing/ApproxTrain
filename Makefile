CC       = gcc
CXX      = g++
CPPFLAGS += -DGOOGLE_CUDA=1 -I.
CFLAGS   += -g -Wall -O2 -std=c++11  -fPIC
LDFLAGS  +=

TF_CFLAGS := $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
CPPFLAGS += $(TF_CFLAGS)
LDFLAGS += $(TF_LFLAGS)

BINARY = convam_gpu.so
OBJ = convam.o

CUDA_ROOT = /usr/local/cuda
CUDA_LIB ?= $(CUDA_ROOT)/lib64
CUDA_OBJ = cuda_kernel.cu.o  gemm.cu.o reverseNswapdim23.cu.o
NVCC ?= nvcc
CUDA_CFLAGS += -g  -O2 -std=c++11 $(CUDA_ARCH) -Xcompiler -Wall -Xcompiler -fPIC  -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr
CUDA_LDFLAGS = -L$(CUDA_LIB) -lcudart
OBJ += $(CUDA_OBJ)

#MULTIPLIER="-DFMBM32_MULTIPLIER=1"
#MULTIPLIER="-DFMBM16_MULTIPLIER=1"

ifeq  ($(MULTIPLIER),)
    MULTIPLIER_CPPFLAG =
else	
	MULTIPLIER_CPPFLAG = -D $(MULTIPLIER)=1
endif

.PHONY: clean test


all: $(BINARY)


$(BINARY): $(OBJ)
	$(CXX) $(CFLAGS) -shared $(OBJ) $(LDFLAGS) $(CUDA_LDFLAGS) -o $@

test_bin: $(OBJ)
	$(CXX)  $(CFLAGS) $(CPPFLAGS) $(OBJ) test/test.cpp $(LDFLAGS) $(CUDA_LDFLAGS) -o $@

convam.o: convam.cc convam.h
	$(CXX) $(CFLAGS) $(CPPFLAGS) $(MULTIPLIER_CPPFLAG) $< -c -o $@


# cuda stuff

cuda_kernel.cu.o: cuda/cuda_kernel.cu cuda/gpu_kernel_helper.h cuda/error.cuh cuda/gemm.cuh cuda/reverseNswapdim23.cuh
	$(NVCC) -x cu $(CUDA_CFLAGS) $(CPPFLAGS) --expt-relaxed-constexpr -c $< -o $@

gemm.cu.o: cuda/gemm.cu
	$(NVCC) -x cu $(CUDA_CFLAGS) $(CPPFLAGS) $(MULTIPLIER_CPPFLAG) -c $< -o $@

reverseNswapdim23.cu.o: cuda/reverseNswapdim23.cu
	$(NVCC) -x cu $(CUDA_CFLAGS) $(CPPFLAGS) -c $< -o $@


clean:
	rm -f *.o *.so

test: $(BINARY)
	python3 mnist_example.py
