# TF2.5
## Installation
### update pip
pip3 install --upgrade pip

### Tensorflow gpu 1.12
pip3 install --user tensorflow

### cuDNN 810 CUDA11.2
## TF2.5 Profile
### resnet50 arch

### TF2.5 ResNet50 Official
100 steps = 28 seconds
114 examples/second
0.28 second/step
### TF2.5 ResNet50 without AM
100 steps =  140 seconds
23 examples/second
1.4 second/step
### TF2.5 ResNet50 with AM
100 steps = 855 seconds
3.74 examples/second
8.55 second/step
### breakdown without AM per step
stage | operation | Time 
------------ | ------------ | ------------- 
Forward | Im2col | 0.05s
Forward | GEMM | 0.274s
BackwardError | Im2Col | 0.091s
BackwardError | GEMM | 0.345s
BackwardFilter | Im2Col | 0.269s
BackwardFilter | GEMM | 0.274s
Total | - | 1.31s
### breakdown with AM per step
stage | operation | Time 
------------ | ------------ | ------------- 
Forward | Im2col | 0.05s
Forward | GEMM | 2.13s
BackwardError | Im2Col | 0.09s
BackwardError | GEMM | 3.05s
BackwardFilter | Im2Col | 0.282s
BackwardFilter | GEMM | 2.237s
Total | - | 7.87s 
## Mnist profile

BATCH Size | Official | No AM | AM
------------ | ------------ | ------------- | ------------- 
1024 | 21ms/step 1s/EPOCH| 128ms/step 9s/EPOCH | 444ms/step 27s/EPOCH
16 | 2ms/step 8s/EPOCH| 4ms/step 15s/EPOCH | 9ms/step 34s/EPOCH

## Imagenet Evaluation
### imagenet run tfrecords preparation scripts

Remember copy all files from valdiation folder to train folder in tf_records folder.

### Clone tf official models repo
https://github.com/tensorflow/models/blob/master/official/README.md

1. Clone the GitHub repository:

```shell
git clone https://github.com/tensorflow/models.git
```

2. Add the top-level ***/models*** folder to the Python path.

```shell
export PYTHONPATH=$PYTHONPATH:/path/to/models
```

If you are using a Colab notebook, please set the Python path with os.environ.

```python
import os
os.environ['PYTHONPATH'] += ":/path/to/models"
```

3. Install other dependencies

```shell
pip3 install --user -r official/requirements.txt
```
### Run
```
export PYTHONPATH=$PYTHONPATH:~/projectAMDNN/models && \
python3 models/official/vision/image_classification/classifier_trainer.py   --mode=train_and_eval   --model_type=resnet   --dataset=imagenet   --model_dir=/home/jing/res50imagenet_test   --data_dir=/home/jing/tfrecordsImagenet/gcs_imagenet/tf_records/train   --config_file=/home/jing/projectAMDNN/models/official/vision/image_classification/configs/examples/resnet/imagenet/gpu.yaml 
```
## Final Test Accuracy

Final Acc/Ref | ACC32 | ACC16 | AM32 | AM16 | FP16
------------ | ------------- | ------------- | ------------- | ------------- | -------------
LENET-300-100 | 96.9% |96.7% | 96.3% |96.8% | Porting
LENET-5 | 98.3% | 98.3% | 98.3% | 98.3% | Porting
RESNET18 |93.22%|93.48%|93.02%|93.40% | Porting
RESNET34 |93.51%|93.73%|93.57%|93.85% | Porting
RESNET50 |93.54%|93.29%|92.89%|93.62% | Porting


