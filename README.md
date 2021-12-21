# AMDNN

<what this is>

## Installation of Dependencies   

AMDNN requires Tensorflow, CUDA Toolkit, cuDNN, GNU C++ compiler and Python3. We recommend using Python 3.5-3.8 and g++ >= 5.4.0.
We did the development and testing on a Ubuntu 18.04.6 environment with Tensorflow 2.3.0, CUDA 10.1, CuDNN 7.6.5, g++ 8.4 and python 3.6.9. A brief guide on installing those dependency versions on an Ubuntu system are given below. ALternatively, If you can follow the official  TensorFlow [build guide](https://www.tensorflow.org/install/source).


### Tensorflow 2.3
    
```
pip3 install --user Tensorflow==2.3.0
```
   
### g++ 8

```
# install g++ version 8
sudo apt -y install  g++8 

# add g++ 8 as an alternative g++
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# update default g++ to version    
# Note: you must select the g++-8 from drop down
sudo update-alternatives --config g++

```    

Note: Make sure your g++ version is greater than 5.4. Otherwise it may lead to a segmentation fault due to a known issue in [Tensorflow]().
    
    
### CUDA Toolkit 10.1

Download CUDA 10.1 from [CUDA 10.1 archive](https://developer.nvidia.com/cuda-10.1-download-archive-base) and follow the steps in the CUDA documentation.
    
### cuDNN 7.6.5

cuDNN is required for other tensorflow official layers, e.g., pooling.

Download CuDNN from the [NVIDIA website](https://developer.nvidia.com/cudnn). Note that you will have to register for this. After downloading the tarball:

```
// decompress cuDNN
tar -xvf cudnn-10.1-linux-x64-v7.6.5.32.tar.xz
// copy to CUDA tool kit directory
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

Further details and alternative installation methods can be found [installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).


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

Select multiplier you would like to test with and select it corresponding flags from  `AMDNN/cuda/gemm.cu` to make

For example,

- In `AMDNN/cuda/gemm.cu`, find a multiplier you would like to use
    

```
#ifdef FPMBM32_MULTIPLIER
...
#endif
```
    
- make

```
make clean && make MULTIPLIER=FMBM32_MULTIPLIER
```
    
- Avaliable Multipliers Name
```
FMBM32_MULTIPLIER
FMBM16_MULTIPLIER
FMBM14_MULTIPLIER
FMBM12_MULTIPLIER
FMBM10_MULTIPLIER
MITCHEL32_MULTIPLIER
MITCHEL16_MULTIPLIER
MITCHEL14_MULTIPLIER
MITCHEL12_MULTIPLIER
MITCHEL10_MULTIPLIER
```
    
### Adding Your Multipliers

Multiplier should take two `float32` as input and output one `float32` if you would like to train/inference.

For inference, it supports integer type.

- In `AMDNN/cuda/gemm.cu`, define `MULTIPLY(a, b)` using your multiplier.

- Add `your-multiplier-design.inl` in `cuda` directory.

- Add `__device__` attribute to all related multiplier function in `your-multiplier-design.inl`.

- make

```
make clean && make MULTIPLIER=YOUR_MULTIPLIER
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

**P.S.** If you would like to use checkpoint between `Conv2D` and `AMConv2D`, make sure you save weights and load weights only. Save/load model will not give your correct graph.(See it on [tensorflow save_and_load](https://www.tensorflow.org/tutorials/keras/save_and_load))

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
    └── _convam_grad.py # gradient register for primitive test
```
    
## Docker Image

TODO

## Troubleshooting

TODO

## Acknowledgement
Mitchell logarithm-based approximate multiplier: [Computer Multiplication and Division Using Binary Logarithms](https://ieeexplore.ieee.org/document/5219391)

Minimally biased multipliers: [Minimally Biased Multipliers for Approximate Integer and Floating-Point Multiplication](https://ieeexplore.ieee.org/document/5219391)

Some code snippets have been taken from [tfapprox](https://github.com/ehw-fit/tf-approximate), [add custom operand to tensorflow](https://github.com/tensorflow/custom-op) and [tensorflow](https://github.com/tensorflow/tensorflow).
