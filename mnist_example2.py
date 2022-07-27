
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
import tensorflow_datasets as tfds
import sys
import tempfile
import numpy as np
from python.keras.layers.amdenselayer import denseam
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mul", type=str, help="multiplier", default="lut/MBM_7.bin")
parser.add_argument("--approx", type=bool, help="use approximate multiplier or not", default=False)
args = parser.parse_args()
APPROX = args.approx
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1 )),
        tf.keras.layers.Flatten(),
denseam(120, activation='relu', mant_mul_lut=args.mul) if APPROX else tf.keras.layers.Dense(120, activation='relu'),
denseam(84, activation='relu', mant_mul_lut=args.mul) if APPROX else tf.keras.layers.Dense(120, activation='relu'),
denseam(10, activation='softmax', mant_mul_lut=args.mul) if APPROX else tf.keras.layers.Dense(120, activation='relu'),
    ])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
model.fit(
    ds_train,
    epochs=20,
    validation_data=ds_test,
)
_, baseline_model_accuracy = model.evaluate(ds_test, verbose=0)
print('Baseline test accuracy:', baseline_model_accuracy)
