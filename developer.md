# Developer Guide    

## Project Structure
```
.
└── ApproxTrain
    ├── cuda 
        ├── cuda_kernel.cu          # include im2col and GPU implementation for OpFunctor (see definition on create custom op https://www.tensorflow.org/guide/create_op)
        ├── gemm.cu                 # GEMM kernel and multiplier definition
        ├── AMsimulator.inl         # LUT-based approximate FP multipliers simulator device function
        ├── approx_mul_lut.cu       # CUDA implementation of LUT
        ├── reverseNswapdim23.cu    # a helper kernel for backpropagation
        ├── *.inl                   # direct c simulated multiplier for 32-bit
        └── other files             # helper function taken from official Tensorflow
    ├── python                      # python wrapper for AMConv2D and AMDense, code snippet taken from official Tensorflow
    ├── test                        # various tests for functional correctness checking
        ├── _convam_grad.py         # gradient register for primitive test
        └── convam_final_test.py    # a primitive test for Convam has exact behavior as Conv2D
    ├── lut
        ├── *.inl                   # direct c simulated multiplier for total bit-width 16-bit or less
        ├── lut_gen.cc              # lookup table generation
        └── lut_gen.sh              # lookup table generation script
    ├── profile                     # benchmark tools  
    ├── convam.cc                   # OpKernel definition AMConv2D
    ├── convam.h                    # OpFunctor definition and partial specialisation AMConv2D
    ├── denseam.cc                  # OpKernel definition AMDense
    ├── denseam.h                   # OpFunctor definition and partial specialisation AMDense
    ├── approx_mul_lut.h            # CPU implemention of LUT
    ├── mnist_example.py            # quick test
    ├── mnist_prunning_example.py   # mnist prunning example
    ├── prunning_script.sh          # script to run prunning
    └── prunning_plotting.py        # plot pruning result
        
        
```
        
    
## Adding  your own approximate multiplier
    
Our in-built multipliers are implemnted in the following files located inside `lut` directory.
    
```
FPmultMBM_fast10.inl 
FPmultMBM_fast12.inl
FPmultMBM_fast14.inl
FPmultMBM_fast16.inl                                                              

Mitchell_10.inl
Mitchell_12.inl
Mitchell_14.inl
Mitchell_16.inl

```
`FPmultMBM_fast*`represents minimally biased multipliers and `Mitchell_*` represents Mitchell logarithm-based approximate multiplier. Numbers following after multiplier's name represent the number of bits of input. Followings are bits format.

| sign | exponent | mantissa | number of total bits |
| ----------- | ----------- | ----------- | ----------- |
| 1 | 8 | 1 | 10 |
| 1 | 8 | 3 | 12 |
| 1 | 8 | 5 | 14 |
| 1 | 8 | 7 | 16 |

    

- Create a similar .inl file (for example `your-multiplier-design.inl`) for your multiplier in lut directory. Multiplier should take two `float` as input and output one `float` if you would like to train and inference. 

- add YOUR_MULTIPLIER_NAME in MULTIPLIER array in lut_gen.sh file.

- add the following code snipet in lut_gen.cc file.
   
```
#elif YOUR_MULTIPLIER_NAME
       #define MULTIPLY(a,b) your_multiplier_function((a),(b));
       #include "your-multiplier-design.inl"
       // define MANTISSA_BITWIDTH of your multiplier, for example 7
       #define MANTISSA_BITWIDTH 7
       // define lookup table binary file name, note this file should always end with postfix "_MANTISSBITWIDTH.bin" to let ApproxTrain generate LUT properly. In this example, the postfix is "_7.bin".
       std::string lut_save = "SOMENAME_7.bin"
```
   
- generate LUT

```
cd lut
./lut_gen.sh
```
    
  

## Use in Tensorflow model

`convam_gpu.so` and "denseam_gpu.so" have been abstracted by python wrapper. It has exact args definition as official Conv2D and Dense respectively, plus mant_mul_lut args. However, we only support input format (batch, img_height, img_width, channel).

```
tf.keras.layers.Conv2D
```

To use approximate Conv2D, import the library

```
from python.keras.layers.am_convolutional import AMConv2D
```

Then you can replace exisiting `Conv2D` with `AMConv2D` in your model definition. For example,

```
# change Conv2D to AMConv2D and use minimally-biased multiplier 16-bit (MBM_7.bin).
#tf.keras.layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu')

AMConv2D(filters=32, kernel_size=5, padding='same', activation='relu', mant_mul_lut="lut/MBM_7.bin")
```


**P.S.** If you would like to use checkpoint between `Conv2D` and `AMConv2D`, make sure you save weights and load weights only. Save/load model will not give your correct graph.(See it on [tensorflow save_and_load](https://www.tensorflow.org/tutorials/keras/save_and_load))
