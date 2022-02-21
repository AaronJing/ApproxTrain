TF_CFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
#MULTIPLIER="-DFMBM16_MULTIPLIER=1"
#MULTIPLIER="-DBFLOAT=1"
nvcc -std=c++11 -c -o denseam.cu.o cuda/denseam.cu \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 $MULTIPLIER -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

g++ -std=c++11 -shared -o denseam.so denseam.cc   denseam.cu.o ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -fPIC -L/usr/local/cuda/lib64 -lcudart ${TF_LFLAGS[@]}
