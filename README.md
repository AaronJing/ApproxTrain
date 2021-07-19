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
### update pip
pip3 install --upgrade pip

### Tensorflow gpu 1.12
pip3 install --user tensorflow-gpu==1.12

### keras
pip3 --user install keras==2.2.0
### cuDNN v7.1.4. cuda9.0
https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v7.1.4/prod/9.0_20180516/cudnn-9.0-linux-x64-v7.1

## Profile command
python3 main.py --phase train --dataset cifar10 --epoch 200 --batch_size 128 --res_n 18 --acc TEST > output

## Profile (no AM)
stage | operation | Time 
------------ | ------------ | ------------- 
Forward | Im2col | 0.0362 s
Forward | GEMM | 0.09 s
BackwardError | Im2Col | 0.0145359s
BackwardError | gemm_inv | 1.69s
BackwardFilter | filtergrad | 2.47s
Total | - | 4.31226s

## Profile (with AM)
stage | operation | Time 
------------ | ------------ | ------------- 
Forward | Im2col | 0.0318 s
Forward | GEMM | 7.6451 s
BackwardError | Im2Col | 0.0147s
BackwardError | gemm_inv | 8.79238s
BackwardFilter | filtergrad | 12.578s
Total | - | 29.0505s

## Profile (no AM -O2 flag)
stage | operation | Time 
------------ | ------------ | ------------- 
Forward | Im2col | 0.0201
Forward | GEMM | 0.044 s
BackwardError | Im2Col | 0.01s
BackwardError | gemm_inv | 0.925s
BackwardFilter | filtergrad | 1.5s
Total | - | 2.5s

## Profile (with AM -O2 flag)
stage | operation | Time 
------------ | ------------ | ------------- 
Forward | Im2col | 0.018 s
Forward | GEMM | 3.695 s
BackwardError | Im2Col | 0.01s
BackwardError | gemm_inv | 4.844s
BackwardFilter | filtergrad | 7.48s
Total | - | 16.0497s

## Profile (no AM -O2 and all im2col+gemm)
stage | operation | Time 
------------ | ------------ | ------------- 
Forward | Im2col | 0.020s
Forward | GEMM | 0.036s
BackwardError | Im2Col | 0.0534s
BackwardError | GEMM | 0.0554s
BackwardFilter | Im2Col | 0.0205s
BackwardFilter | GEMM | 0.047s
Total | - | 0.232s

## Profile (AM -O2 and all im2col+gemm)
stage | operation | Time 
------------ | ------------ | ------------- 
Forward | Im2col | 0.020s
Forward | GEMM | 0.3s
BackwardError | Im2Col | 0.052s
BackwardError | GEMM | 0.335s
BackwardFilter | Im2Col | 0.018s
BackwardFilter | GEMM | 0.37s
Total | - | 1.092s


## Profile (TF)
Total 1.27 s
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
