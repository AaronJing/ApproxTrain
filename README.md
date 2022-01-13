# AMDNN

<what this is>

## Installation of Dependencies   

AMDNN requires Tensorflow, CUDA Toolkit, cuDNN, GNU C++ compiler and Python3. We recommend using Python 3.5-3.8 and g++ 5.4.0 or higher.
We did the development and testing on an Ubuntu 18.04.6 environment with Tensorflow 2.3.0, CUDA 10.1, CuDNN 7.6.5, g++ 8.4 and python 3.6.9. A brief guide on installing those dependency versions on an Ubuntu system are given below. Alternatively, If you can follow the official  TensorFlow [build guide](https://www.tensorflow.org/install/source).


### Tensorflow 2.3
    
```
# check Tensorflow version
python3 -c 'import tensorflow as tf; print(tf.__version__)'

# install tensorflow 2.3.0
pip3 install --user Tensorflow==2.3.0
```
   
### g++ 8

```
# Check g++ version
g++ -v
    
# install g++ version 8
sudo apt -y install  g++-8 

# add g++ 8 as an alternative g++
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8

# update default g++ to version    
# Note: you must select the g++-8 from drop down
sudo update-alternatives --config g++

```    

Note: Make sure your g++ version is greater than 5.4. Otherwise it may lead to a segmentation fault due to a known issue in [Tensorflow]().
    
    
### CUDA Toolkit 10.1

```
# Check CUDA version
nvidia-smi
```     
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


## Running an Example

### Install tensorflow dataset

In our example, we use *tfds* package to load and preprocess the train/test images. Install *tfds* as:

``
pip3 install --user tensorflow-datasets
``

### Clone the repository

```
git clone https://github.com/AaronJing/AMDNN
cd AMDNN
```
    
### Using in-built approximate multipliers
    
    
We provides two approximate multipliers, minimally biased multiplier (MBM) and Mitchell logarithm-based approximate multiplier with different bit-widths as shown in the table below.
    
| Name 						 | Multiplier | Bit-width  |
|----------------------------|------------|------------|
| FMBM32_MULTIPLIER          | MBM        | 32		   |	
| FMBM16_MULTIPLIER          | MBM        | 16         |
| FMBM14_MULTIPLIER          | MBM        | 14         |
| FMBM12_MULTIPLIER          | MBM        | 12         |
| FMBM10_MULTIPLIER          | MBM        | 10         |
| MITCHEL32_MULTIPLIER       | Mitchell   | 32         |
| MITCHEL16_MULTIPLIER       | Mitchell   | 16         |
| MITCHEL14_MULTIPLIER       | Mitchell   | 14         |
| MITCHEL12_MULTIPLIER       | Mitchell   | 12         |
| MITCHEL10_MULTIPLIER       | Mitchell   | 10         |
    
Now build our library using *make* command followed by the name of multiplier. For example, to build with  MBM 32 bit multiplier:

```
make clean && make MULTIPLIER=FMBM32_MULTIPLIER
```
    
If you do not specify a multiplier, i.e., if you just call `make`, the library will be built with IEEE 754 single precision multiplication (* operator).
    
Warnings come from Tensorflow library could be safely ignored.
    
When building is sucessful, `Convam.so` file is created. Now, launch the example script as:
    
```
python3 mnist_example.py    
```    

You would expect 98% accuracy or higher, if everything works properly.
 
    
## For Developers    

If you are interested in adding your own multiplier or your own dataset, please visit the [developers guide](developer.md).
    

## Docker Image

TODO

## Troubleshooting

TODO

## Acknowledgement
Mitchell logarithm-based approximate multiplier: [Computer Multiplication and Division Using Binary Logarithms](https://ieeexplore.ieee.org/document/5219391)

Minimally biased multipliers: [Minimally Biased Multipliers for Approximate Integer and Floating-Point Multiplication](https://ieeexplore.ieee.org/document/5219391)

Some code snippets have been taken from [tfapprox](https://github.com/ehw-fit/tf-approximate), [add custom operand to tensorflow](https://github.com/tensorflow/custom-op) and [tensorflow](https://github.com/tensorflow/tensorflow).
