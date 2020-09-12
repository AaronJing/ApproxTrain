# AMDNN
## Prerequirement
- [x] Identical output with tensorflow official implementation will ensure the correctness of our implementation.
## TODO LIST
- [x] Set up rollback. (Even program crashed, we could start from last checkpoint)
- [x] Set up walltime limit 48 hours. Every two days, the partial results will be transferred to storage.
- [ ] Mask floating point for Bfloat16 after multiplication.
- [x] T_SIZE should be 0 or 16 bits.



Model | TF | ACC32 | ACC16 | AM32 | AM16 | FP16
------------ | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
LENET-300-100 |<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>
LENET-5 |<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>
RESNET18 |<ul><li>- [x] </li>|<ul><li>- [x] </li>|<ul><li>- [x] </li>|<ul><li>- [x] </li>|<ul><li>- [x] </li>|<ul><li>- [ ] </li>
RESNET34 |<ul><li>- [x] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>
RESNET50 |<ul><li>- [x] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>| <ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>

## Final Test Accuracy

Final Acc/Ref | TF | ACC32 | ACC16 | AM32 | AM16 | FP16
------------ | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
LENET-300-100 ||||||
LENET-5 ||||||
RESNET18 |93.48%/91.25 %|93.55%||||
RESNET34 |93.67%/92.49 %|||||
RESNET50 |93.66%/92.83 %|||||

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
