# AMDNN

## Dependency

### Tensorflow

Our code has been tested with Tensorflow 2.3.0

For Ubuntu user,

``
pip3 install --user Tensorflow==2.3.0
``

### CUDA & GCC/G++

Our code base is built against CUDA 10.1 and gcc/g++ 8.4.0.

Download CUDA10.1 from [CUDA 10.1 archive](https://developer.nvidia.com/cuda-10.1-download-archive-base).

Other gcc/g++ versions might lead errors.

For Ubuntu user,

```

sudo apt -y install gcc-8 g++8

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 8 

sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

// you should be able to select the gcc/g++ 8 from drop down

sudo update-alternatives --config gcc

sudo update-alternatives --config g++

```

To make your program run properly, cuDNN is required as well.
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

In `AMDNN/cuda/gemm.cu`, find a multiplier you would like to use

```

#ifdef FPMBM32_MULTIPLIER
...
#endif
```


In `AMDNN/gpu_compile.sh`, define it as following

```
MULTIPLIER="-DFMBM32_MULTIPLIER=1"
```

### Adding Your Multipliers

## Project Structure

## Docker Image

TODO

## Acknowledgement
Mitchell logarithm-based approximate multiplier: [Computer Multiplication and Division Using Binary Logarithms](https://ieeexplore.ieee.org/document/5219391)

Minimally biased multipliers: [Minimally Biased Multipliers for Approximate Integer and Floating-Point Multiplication](https://ieeexplore.ieee.org/document/5219391)

Some code snippets have been taken from [tfapprox](https://github.com/ehw-fit/tf-approximate), [add custom operand to tensorflow](https://github.com/tensorflow/custom-op) and [tensorflow](https://github.com/tensorflow/tensorflow).
