# AMDNN

## Dependency

### Tensorflow

Our code has been tested with Tensorflow 2.3.0

For Ubuntu user,

``
pip3 install --user Tensorflow==2.3.0
``

### Python3

We recommend using python3 3.5-3.8

### CUDA 
Our code base is built against CUDA 10.1

Download CUDA10.1 from [CUDA 10.1 archive](https://developer.nvidia.com/cuda-10.1-download-archive-base).
### GCC/G++
gcc/g++ 8.4.0.

Other gcc/g++ versions might lead errors.

For Ubuntu user,

```
// install g++_8

sudo apt -y install  g++8

// give g++-8 priority

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

// you should be able to select the g++-8 from drop down

sudo update-alternatives --config g++

```
### cuDNN

cuDNN is reuiqred for other tensorflow official layers, e.g., pooling.

We use cuDNN 7.6.5.

After downloading, 

```
// decompress cuDNN
tar -xvf cudnn-10.1-linux-x64-v7.6.5.32.tar.xz
// copy to CUDA tool kit directory
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```
Details can be found [installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).

### alternative version

If you do not like to use provided dependency, you can follow official [build guide](https://www.tensorflow.org/install/source).

## Example

### Install tensorflow dataset

In our example, we use tfds to pipeline the train/test image.

For Ubuntu user,

``
pip3 install --user tensorflow-datasets
``

### Using Existing Multipliers

We provides two approximate multipliers, Mitchell logarithm-based approximate multiplier and minimally biased multipliers in `cuda` directory.
```
FPmultMBM_fast10.inl
FPmultMBM_fast12.inl
FPmultMBM_fast14.inl
FPmultMBM_fast16.inl                                                              
FPmultMBM_fast32.inl
Mitchell_10.inl
Mitchell_12.inl
Mitchell_14.inl
Mitchell_16.inl
Mitchell_32.inl
```
`FPmultMBM_fast*`represents minimally biased multipliers and `Mitchell_*` represents Mitchell logarithm-based approximate multiplier. Numbers following after multiplier's name represent the number of bits of input. Followings are bits format.

| sign | exponent | mantissa | number of total bits |
| ----------- | ----------- | ----------- | ----------- |
| 1 | 8 | 1 | 10 |
| 1 | 8 | 3 | 12 |
| 1 | 8 | 5 | 14 |
| 1 | 8 | 7 | 16 |
| 1 | 8 | 23 | 32 |

Select multiplier you would like to test with and select it corresponding flags from  `AMDNN/cuda/gemm.cu` to define in `gpu_compile.sh`.

For example,

- In `AMDNN/cuda/gemm.cu`, find a multiplier you would like to use

```
#ifdef FPMBM32_MULTIPLIER
...
#endif
```


- In `AMDNN/gpu_compile.sh`, define it as following

```
MULTIPLIER="-DFMBM32_MULTIPLIER=1"
```

- Compile into library

```
./gpu_compile.sh
```
### Adding Your Multipliers

Multiplier should take two `float32` as input and output one `float32` if you would like to train/inference.

For inference, it supports integer type.

- In `AMDNN/cuda/gemm.cu`, define `MULTIPLY(a, b)` using your multiplier.

- Add `your-multiplier-design.inl` in `cuda` directory.

- Add `__device__` attribute to all related multiplier function in `your-multiplier-design.inl`.

- In `AMDNN/gpu_compile.sh`, define it as following

```
MULTIPLIER="-DYOUR_MULTIPLIER=1"
```

- Compile into library `convam.so`

```
./gpu_compile.sh
```

## Use in Tensorflow model

`Convam.so` has been abstracted by python wrapper. It has exact args definition as official Conv2D. However, we only support input format (batch, img_height, img_width, channel). Dilation is not supported at the momemnt. 

```
tf.keras.layers.Conv2D
```

To use approximate Conv2D, import the library

```
from python.keras.layers.am_convolutional import AMConv2D
```

Then you can replace exisiting `Conv2D` with `AMConv2D` in your model definition.

**P.S.** If you would like to use checkpoint between `Conv2D` and `AMConv2D`, make sure you save weights and load weights only. Save/load model might not give your correct graph.(See it on [tensorflow save_and_load](https://www.tensorflow.org/tutorials/keras/save_and_load))

## Project Structure
```
.
└── AMDNN
    └── python # python wrapper for AMConv2D, code snippet taken from official Tensorflow
    └── cuda
        ├── cuda_kernel.cu # include im2col and GPU implementation for OpFunctor
        ├── gemm.cu # GEMM kernel and multiplier definition
        ├── reverseNswapdim23.cu # a helper kernel for backpropagation
        ├── *.inl # multiplier
        └── other files # helper function taken from official Tensorflow
    └── example
    └── Convam.cc # OpKernel definition
    └── Convam.h # OpFunctor definition and partial specialisation
    └── convam_final_test.py # a primitive test for Convam has exact behavior as Conv2D
    └── _convam_grad.py # gradient register for above primitive test

```
## Docker Image

TODO

## Troubleshooting

TODO

## Acknowledgement
Mitchell logarithm-based approximate multiplier: [Computer Multiplication and Division Using Binary Logarithms](https://ieeexplore.ieee.org/document/5219391)

Minimally biased multipliers: [Minimally Biased Multipliers for Approximate Integer and Floating-Point Multiplication](https://ieeexplore.ieee.org/document/5219391)

Some code snippets have been taken from [tfapprox](https://github.com/ehw-fit/tf-approximate), [add custom operand to tensorflow](https://github.com/tensorflow/custom-op) and [tensorflow](https://github.com/tensorflow/tensorflow).
