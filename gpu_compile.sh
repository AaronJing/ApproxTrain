TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc -std=c++11 -c -o cuda_kernel.cu.o cuda/cuda_kernel.cu \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o convam_gpu.so convam.cc   cuda_kernel.cu.o ${TF_CFLAGS[@]}  -fPIC -L/usr/local/cuda-9.0/lib64 -lcudart ${TF_LFLAGS[@]}
# TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
# TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

# nvcc -std=c++11 -c -o cuda_kernel_improved.cu.o cuda_kernel_improved.cu \
#   ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# g++ -std=c++11 -shared -o convam_gpu.so convam.cc   cuda_kernel_improved.cu.o ${TF_CFLAGS[@]}  -fPIC -L/usr/local/cuda-9.0/lib64 -lcudart ${TF_LFLAGS[@]}
