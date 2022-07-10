import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
#tf.random.set_seed(1234)
tf.get_logger().setLevel("ERROR")
import tensorflow_datasets as tfds
import sys
import tempfile
import numpy as np
from python.keras.layers.am_convolutional import AMConv2D
from python.keras.layers.amdenselayer import denseam
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mul", type=str, help="multiplier", default="MBM16")
parser.add_argument("--approx", type=bool, help="use approximate multiplier or not", default=False)
args = parser.parse_args()
def get_gzipped_model_size(file):
  # Returns size of gzipped model, in bytes.
  import zipfile

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(file)

  return os.path.getsize(zipped_file)
APPROX = args.approx
conv2d = AMConv2D if APPROX else tf.keras.layers.Conv2D
dense = denseam if APPROX else tf.keras.layers.Dense
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
 tf.keras.layers.Input(shape=(28, 28, 1)),
 conv2d(filters=6, kernel_size=5, padding='same', activation='relu'),
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'),
 conv2d(filters=16, kernel_size=5, padding='same', activation='relu'),
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
 tf.keras.layers.Flatten(),
 dense(120, activation='relu'),
 dense(84, activation='relu'),
 dense(10, activation='softmax')
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
modeldir = "./checkpoint/" + args.mul+'.h5'
tf.keras.models.save_model(model, modeldir, include_optimizer=False)
print('Saved baseline model to:', modeldir)

import tensorflow_model_optimization as tfmot
# read final sparisty from files
with open("final_sparsity.txt", "r") as file1:
    for line in file1.readlines():
        f_sparsity_list = [float(i) for i in line.split(",")]

for fs in f_sparsity_list:
    p_model = tf.keras.models.clone_model(model)
    p_model.set_weights(model.get_weights())
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    batch_size = 128
    epochs = 2


    num_batch = ds_train.cardinality().numpy()
    end_step = np.ceil(num_batch).astype(np.int32) * epochs

    # Define model for pruning.
    pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,
                                                               final_sparsity=fs,
                                                               begin_step=0,
                                                               end_step=end_step)
    }

    model_for_pruning = prune_low_magnitude(p_model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    model_for_pruning.summary()

    dirname = args.mul
    logdir = "./logs/" + dirname + str(fs)


    callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
    ]

    model_for_pruning.fit(ds_train,
                  batch_size=batch_size, epochs=epochs,
                  callbacks=callbacks)
    for layer in model_for_pruning.layers:
        print("Layer Name: ", layer.name)
        flat_weight = layer.get_weights()[0].flat
        flat_bias = layer.get_weights()[1].flat
        zeronum = 0
        totalnum = 0
        for i in flat_weight:
            if i == 0.0:
                zeronum += 1
            totalnum += 1
        for i in flat_bias:
            if i == 0.0:
                zeronum += 1
            totalnum += 1
        print("Total Weight Number: ", totalnum, "Zero Number: ", zeronum, "Sparsity: ", zeronum/totalnum)
    _, model_for_pruning_accuracy = model_for_pruning.evaluate(ds_test, verbose=0)
    print('Model sparsity:', fs)
    print('Baseline test accuracy:', baseline_model_accuracy) 
    print('Pruned test accuracy:', model_for_pruning_accuracy)
    print('Logdir:', logdir)
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    _, pruned_keras_file = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(model_for_export, pruned_keras_file, include_optimizer=False)
    print('Saved pruned Keras model to:', pruned_keras_file)
    converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
    converter.allow_custom_ops = True 
    pruned_tflite_model = converter.convert()
    _, pruned_tflite_file = tempfile.mkstemp('.tflite')
    with open(pruned_tflite_file, 'wb') as f:
        f.write(pruned_tflite_model)
    print('Saved pruned TFLite model to:', pruned_tflite_file)
    print("Size of gzipped baseline Keras model: %.2f bytes" % (get_gzipped_model_size(modeldir)))
    print("Size of gzipped pruned Keras model: %.2f bytes" % (get_gzipped_model_size(pruned_keras_file)))
    print("Size of gzipped pruned TFlite model: %.2f bytes" % (get_gzipped_model_size(pruned_tflite_file)))
