# AMDNN
## Project Structure
* **_convam_grad.py** register gradient operations for CPU version of convam
* **_convam_grad_gpu.py** register gradient operations for GPU version of convam
* **convam.cc** size calculation, memory allocation, preprocessing(flatten neurons) and call kernel
* **cuda_kernel.cu** all gpu kernels and AM emulators here
	* *gemm(...)* shared memory version. It is used in forward propagation
	* *im2col_improved(...)* Used in forward propagation
	* *filtergradkernel(...)* Used in backpropagation: filtergradient
	* *im2col(...)* Used in backpropagation: Error
	* *inserthole* Used in backpropagation: Error
	* *gemm_inverse(...)* Used in backpropagation Error
* **gpu_compile.sh** GPU compliation (it also compiles CPU)
* **finally.pbs** gadi script example

## Installation

### Tensorflow gpu 1.12
pip3 install --upgrade pip
### keras
pip3 --user install keras==2.2.0


## Final Test Accuracy

Final Acc/Ref | ACC32 | ACC16 | AM32 | AM16 | FP16
------------ | ------------- | ------------- | ------------- | ------------- | -------------
LENET-300-100 | 96.9% |96.7% | 96.3% |96.8% | Porting
LENET-5 | 98.3% | 98.3% | 98.3% | 98.3% | Porting
RESNET18 |93.22%|93.48%|93.02%|93.40% | Porting
RESNET34 |93.51%|93.73%|93.57%|93.85% | Porting
RESNET50 |93.54%|93.29%|92.89%|93.62% | Porting

## Files directory
### Scylla.home:
## Compile Kernel Cuda on NCI Gadi
```
qsub -I -limage=raijin
module  unload intel-mkl/2019.2.187
module load tensorflow/1.12-cudnn7.1-py36
./gpu_compiler
```
## NCI information
New /scratch space - Very large (!) quotas, but files will be automatically deleted after 90 days
## NCI error
5:12AM 15/08/2020 Resnet34-AM32 Bus error at the epoch 87, batch 43 and restart
5:12AM 15/08/2020 Resnet34-AM16 Bus error at the beginning. 
5:12AM 15/08/2020 Resnet50-AM16 Bus error at the beginning.
