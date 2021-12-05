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

### Adding Your Multipliers

## Project Structure

## Docker Image

TODO


