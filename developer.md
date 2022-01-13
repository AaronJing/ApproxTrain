# Developer Guide    

## Project Structure
```
.
└── AMDNN
    └── cuda 
        ├── cuda_kernel.cu          # include im2col and GPU implementation for OpFunctor (see definition on create custom op https://www.tensorflow.org/guide/create_op)
        ├── gemm.cu                 # GEMM kernel and multiplier definition
        ├── reverseNswapdim23.cu    # a helper kernel for backpropagation
        ├── *.inl                   # multiplier implementations
        └── other files             # helper function taken from official Tensorflow
    └── python                      # python wrapper for AMConv2D, code snippet taken from official Tensorflow
    └── test                        # various tests for functional correctness checking
        ├── _convam_grad.py         # gradient register for primitive test
        └── convam_final_test.py    # a primitive test for Convam has exact behavior as Conv2D           
    └── convam.cc                   # OpKernel definition
    └── convam.h                    # OpFunctor definition and partial specialisation
    └── mnist_example.py            # quick test
        
        
```
        
    
## Adding  your own approximate multiplier
    
Our in-built multipliers are implemnted in the following files located inside `cuda` directory.
    
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
    

- Create a similar .inl file (for example `your-multiplier-design.inl`) for your multiplier in cuda directory. Multiplier should take two `float` as input and output one `float` if you would like to train/inference. For inference, it supports integer type. 
   
- Make sure you add `__device__` attribute to all related multiplier functions in `your-multiplier-design.inl`.    

- In `cuda/gemm.cu`, add an #elif block thatt defines `MULTIPLY(a, b)` using your multiplier. For example, add the following just after `#include "Mitchell_12.inl"` and before `#else`.

 
```
    #elif YOUR_MULTIPLIER
       #define MULTIPLY(a,b) your_multiplier_function((a),(b));
       #include "your-multiplier-design.inl"
```
    
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
