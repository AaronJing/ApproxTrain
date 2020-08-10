# AMDNN
## TODO LIST

Model | ACC32 | ACC16 | AM32 | AM16 
------------ | ------------- | ------------- | ------------- | -------------
LENET-300-100 |<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>
LENET-5 |<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>
RESNET18 |<ul><li>- [x] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>
RESNET34 |<ul><li>- [x] </li> |<ul><li>- [ ] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>
RESNET50 |<ul><li>- [x] </li>|<ul><li>- [ ] </li>|<ul><li>- [ ] </li>| <ul><li>- [ ] </li>
  
## Final Test Accuracy

Final Acc/Ref | ACC32 | ACC16 | AM32 | AM16 
------------ | ------------- | ------------- | ------------- | -------------
LENET-300-100 ||||
LENET-5 ||||
RESNET18 |93.48%|||
RESNET34 |93.67%|||
RESNET50 |93.66%|||

## Files directory
### Scylla.home:
## Compile Kernel Cuda on NCI Gadi
```
qsub -I -limage=raijin
module  unload intel-mkl/2019.2.187
module load tensorflow/1.12-cudnn7.1-py36
./gpu_compiler
```
