import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from python.keras.layers.amdenselayer import denseam
from python.keras.layers.am_convolutional import AMConv2D
LUT = 'lut/MBM_7.bin'


(x_train, y_train), (x_test, y_test) = tfds.as_numpy(tfds.load(
    'cifar10', split=['train', 'test'], batch_size=-1, as_supervised=True))
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

model = tf.keras.Sequential([
    AMConv2D(32, (3, 3), activation='relu', mant_mul_lut=LUT),
    AMConv2D(32, (3, 3), activation='relu', mant_mul_lut=LUT),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    AMConv2D(64, (3, 3), activation='relu', mant_mul_lut=LUT),
    AMConv2D(64, (3, 3), activation='relu', mant_mul_lut=LUT),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    denseam(512, activation='relu', mant_mul_lut=LUT),
    tf.keras.layers.Dropout(0.5),
    denseam(10, activation='softmax', mant_mul_lut=LUT),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)
model.fit(x_train, y_train, epochs=20, batch_size=64, validation_split=0.1)
model.summary()
loss, acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {acc:.4f}")
