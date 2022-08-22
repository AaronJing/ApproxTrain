import tensorflow as tf
import numpy as np
import sys
from python.keras.layers.am_convolutional import AMConv2D
from python.keras.layers.amdenselayer import denseam
import os
import utils
import resnet50
import argparse
import datetime

p = argparse.ArgumentParser(description="commands for profiling")
p.add_argument('--model', help="model to profile")
p.add_argument('--batch-size', help="profile batch size")
p.add_argument('--batch-number', help="profile batch number")
p.add_argument('--am', type=bool,help="approximate multiplications?")
p.add_argument('--gpu', help="gpu==1 enable the GPU otherwise CPU")
args = p.parse_args()
test_batch = int(args.batch_number)
batch_size = int(args.batch_size)
AM = True if args.am else False
if int(args.gpu) == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("Model: " + args.model)
print("Batch_size: " + args.batch_size)
print("Batch_number: " + args.batch_number)
print("am: " + str(args.am))
print("gpu: " + args.gpu)
def fake_imagenet_data():
    return np.random.normal(0, 2.5, size=(test_batch*batch_size, 224, 224, 3)), np.random.normal(0, 2.5, size=(test_batch*batch_size, 1000))
def fake_cifar10_data():
    return np.random.normal(0, 2.5, size=(test_batch*batch_size, 32, 32, 3)), np.random.normal(0, 2.5, size=(test_batch*batch_size, 10))
def fake_mnist_data():
    return np.random.normal(0, 2.5, size=(test_batch*batch_size, 28, 28, 1)), np.random.normal(0, 2.5, size=(test_batch*batch_size, 10))
model = None
x_train = None
y_train = None
if args.model == "resnet18":
    model=resnet50.ResNet18(weights=None, AM=AM)
    x_train, y_train = fake_cifar10_data()
elif args.model == "resnet34":
    model=resnet50.ResNet34(weights=None, AM=AM)
    x_train, y_train = fake_cifar10_data()
elif args.model == "resnet50":
    model=resnet50.ResNet50(weights=None, AM=AM)
    x_train, y_train = fake_cifar10_data()
elif args.model == "resnet50ImageNet":
    model=resnet50.ResNet50ImageNet(weights=None, AM=AM)
    x_train, y_train = fake_imagenet_data()
elif args.model == "lenet5":
    model=resnet50.lenet5(AM)
    x_train, y_train = fake_mnist_data()
elif args.model == "lenet31":
    model=resnet50.lenet31(AM)
    x_train, y_train = fake_mnist_data()
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

#model.evaluate(x_test, y_test, verbose=1)
#x_train, y_train = fake_imagenet_data()
#x_test, y_test = fake_imagenet_data()
#model.summary()
#exit(0)
class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        pass
    def on_train_batch_begin(self,batch,logs = {}):
        self.start_time = datetime.datetime.now()
    def on_train_batch_end(self,batch,logs = {}):
        self.elapsed = datetime.datetime.now() - self.start_time
        print("")
        print(args.model," elapsed time: ", int(self.elapsed.total_seconds()*1000000), " batch num ", batch)
        print("")
    def on_test_batch_begin(self,batch,logs = {}):
        self.start_time = datetime.datetime.now()
    def on_test_batch_end(self,batch,logs = {}):
        self.elapsed = datetime.datetime.now() - self.start_time
        print("")
        print(args.model," elapsed time: ", int(self.elapsed.total_seconds()*1000000), " batch num ", batch)
        print("")
mycallback = timecallback()
#model.fit(
        #x_train[:test_batch*batch_size],
        #y_train[:test_batch*batch_size],
        #batch_size=batch_size,
        #epochs=1,
        #callbacks=[mycallback],
        #verbose=0
#)
model.evaluate(x_train[:test_batch*batch_size], y_train[:test_batch*batch_size], batch_size=batch_size, verbose=0, callbacks=[mycallback])
#print(args.am)
