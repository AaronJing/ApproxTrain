# TF2.5
## Installation
### update pip
pip3 install --upgrade pip

### Tensorflow gpu 1.12
pip3 install --user tensorflow

### cuDNN 810 CUDA11.2

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
python3 classifier_trainer.py \
  --mode=train_and_eval \
  --model_type=resnet \
  --dataset=imagenet \
  --model_dir=/home/jing/res50imagenet_test \
  --data_dir=/home/jing/tfrecordsImagenet/gcs_imagenet/tf_records/train \
  --config_file=configs/examples/resnet/imagenet/gpu.yaml
```
## Final Test Accuracy

Final Acc/Ref | ACC32 | ACC16 | AM32 | AM16 | FP16
------------ | ------------- | ------------- | ------------- | ------------- | -------------
LENET-300-100 | 96.9% |96.7% | 96.3% |96.8% | Porting
LENET-5 | 98.3% | 98.3% | 98.3% | 98.3% | Porting
RESNET18 |93.22%|93.48%|93.02%|93.40% | Porting
RESNET34 |93.51%|93.73%|93.57%|93.85% | Porting
RESNET50 |93.54%|93.29%|92.89%|93.62% | Porting


