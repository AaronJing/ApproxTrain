# ApproxTrain

ApproxTrain is an open-source framework that allows sufficiently fast evaluation of DNNs training and inference using simulated approximate multipliers. ApproxTrain is as user-friendly as TensorFlow  (TF) and requires only a high-level description of a DNN architecture along with C/C++ functional models of the approximate multiplier. We improve the speed of the simulation at the multiplier level using a novel LUT-based approximate floating-point (FP) multiplier on GPU (AM Simulator). Additionally, a novel flow is presented to seamlessly convert C/C++ functional models of approximate FP multipliers into AM Simulator. ApproxTrain leverages CUDA and efficiently integrates AM Siumlator into TensorFlow library, to overcome the absence of native hardware approximate multiplier in commercial GPUs.

## Installation of Dependencies   

ApproxTrain requires Tensorflow, CUDA Toolkit, cuDNN, GNU C++ compiler and Python3. We recommend using Python 3.5-3.8 and g++ 5.4.0 or higher.
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
# decompress cuDNN
tar -xvf cudnn-10.1-linux-x64-v7.6.5.32.tar.xz
# copy to CUDA tool kit directory
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

Further details and alternative installation methods can be found [installation guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).

## Supporting operands
| Apporximate Operator | Tensorflow equivalent | Gradient Computation | Status   |
|----------------------|-----------------------|----------------------|----------|
| AMConv2D             | Conv2D                | ✔                    | Complete |
| denseam              | Dense                 | ✔                    | Complete |
| MatMulAM             | MatMul                | ✔                    | Complete |
| AMMHA                | MultiHeadAttention    | ✔                    | Complete |

## Running an Example

### Install tensorflow dataset

In our example, we use *tfds* package to load and preprocess the train/test images. Install *tfds* as:

``
pip3 install --user tensorflow-datasets
``

### Clone the repository

```
git clone https://github.com/AaronJing/ApproxTrain
cd ApproxTrain
```
### Compile the framework with AM Simulator

Warnings come from Tensorflow library could be safely ignored.
    
When building is sucessful, `convam_gpu.so` and 'denseam_gpu.so' file are created.

```
make clean && make convam MULTIPLIER=AMSIMULATOR && make denseam_gpu.so MULTIPLIER=AMSIMULATOR
```

If you do not specify AMSIMULATOR, framework will be built with native hardware multiplication FP32 (* operator).

### Using in-built approximate multipliers
    
    
We provides two approximate multipliers, minimally biased multiplier (MBM) and Mitchell logarithm-based (MIT) approximate multiplier with different bit-widths as shown in the table below. (s, e, m) represents sign, exponent and mantissa. For lookup table-based (lut) AM Simulator, we support mantissa bit-width from 1 (16 Bytes) to 11 (16.8 Mb). For multipliers with mantissa bit-width greater or equal to 12, direct C simulation should be used and expect performance degradation. Note that, the number after underscore in column Generate LUT File Name is the mantissa bit-width, which is needed for ApproxTrain to initialize lookup table.
    
| Name 						 | Multiplier | Bit-width  |(s, e, m)   |LUT AM Simulator   |Generated LUT File Name   |
|----------------------------|------------|------------|------------|------------|------------|
| FMBM32_MULTIPLIER          | MBM        | 32         |(1, 8, 23)  |❌|❌|
| FMBM16_MULTIPLIER          | MBM        | 16         |(1, 8, 7)   |✔|MBM_7.bin|
| FMBM14_MULTIPLIER          | MBM        | 14         |(1, 8, 5)   |✔|MBM_5.bin|
| FMBM12_MULTIPLIER          | MBM        | 12         |(1, 8, 3)   |✔|MBM_3.bin|
| FMBM10_MULTIPLIER          | MBM        | 10         |(1, 8, 1)   |✔|MBM_1.bin|
| MITCHEL23_MULTIPLIER       | MIT  | 32         |(1, 8, 23)  |❌|❌|
| MITCHEL16_MULTIPLIER       | MIT   | 16         |(1, 8, 7)   |✔|MIT_7.bin|
| MITCHEL14_MULTIPLIER       | MIT   | 14         |(1, 8, 5)   |✔|MIT_5.bin|
| MITCHEL12_MULTIPLIER       | MIT   | 12         |(1, 8, 3)   |✔|MIT_3.bin|
| MITCHEL10_MULTIPLIER       | MIT   | 10         |(1, 8, 1)   |✔|MIT_1.bin|

Now generate binary LUT files for AMSimulator

```
cd lut
./lut_gen.sh
```
Now, launch the example script with preferred approximate multiplier (e.g. FMBM16_MULTIPLIER) as:
    
```
python3 mnist_example.py --mul="lut/MBM_7.bin"
```    

You would expect 98% accuracy or higher, if everything works properly.
 
If you are not sure about whether you are running GPU or not, run the following commands.

```
python3 -c 'import tensorflow as tf; print(tf.test.gpu_device_name())'
```
  
## For Developers    

If you are interested in adding your own multiplier or your own dataset, please visit the [developers guide](developer.md).
    
## Troubleshooting

TODO

## Acknowledgement
Mitchell logarithm-based approximate multiplier: [Computer Multiplication and Division Using Binary Logarithms](https://ieeexplore.ieee.org/document/5219391)

Minimally biased multipliers: [Minimally Biased Multipliers for Approximate Integer and Floating-Point Multiplication](https://ieeexplore.ieee.org/document/5219391)

Some code snippets have been taken from [tfapprox](https://github.com/ehw-fit/tf-approximate), [add custom operand to tensorflow](https://github.com/tensorflow/custom-op) and [tensorflow](https://github.com/tensorflow/tensorflow).
