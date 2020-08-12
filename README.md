# AMDNN
## Prerequirement
- [x] Identical output with tensorflow official implementation will ensure the correctness of our implementation.
## TODO LIST
- [x] Set up rollback. (Even program crashed, we could start from last checkpoint)
- [x] Set up walltime limit 48 hours. Every two days, the partial results will be transferred to storage.
- [ ] Mask floating point for Bfloat16 after multiplication.

GPU\Model | Resnet18 acc | Resnet34 acc | Resnet50 acc | Resnet18 am | Resnet34 am | Resnet50 am
------------ | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
GTX1080|||
Gadi(volta)|0.6h/epoch or 7.5days/300epoch|0.92h/epoch or 11.45 days/300epoch|2.38h/epoch or 29.8 days/300epoch|0.73h/epoch or 9.15days/300epoch||
GTX1080 x2 (deep bayes)|||

Model | ACC32 | ACC16 | AM32 | AM16 
------------ | ------------- | ------------- | ------------- | -------------
LENET-300-100 |<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>
LENET-5 |<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>
RESNET18 |<ul><li>- [x] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>
RESNET34 |<ul><li>- [x] </li> |<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>
RESNET50 |<ul><li>- [x] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>| <ul><li>- [ ] </li>
  
## Final Test Accuracy

Final Acc/Ref | TF | ACC32 | ACC16 | AM32 | AM16 | FP16
------------ | ------------- | ------------- | ------------- | ------------- | -------------
LENET-300-100 ||||||
LENET-5 ||||||
RESNET18 |93.48%|||||
RESNET34 |93.67%|||||
RESNET50 |93.66%|||||

## Files directory
### Scylla.home:
## Compile Kernel Cuda on NCI Gadi
```
qsub -I -limage=raijin
module  unload intel-mkl/2019.2.187
module load tensorflow/1.12-cudnn7.1-py36
./gpu_compiler
```
