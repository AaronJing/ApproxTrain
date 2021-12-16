import tensorflow as tf
#import tensorflow_datasets as tfds
import sys
from python.keras.layers.am_convolutional import AMConv2D
model = tf.keras.models.Sequential([
 tf.keras.layers.Input(shape=(28, 28, 1)),
 AMConv2D(filters=32, kernel_size=5, padding='same', activation='relu'),
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'),
 AMConv2D(filters=32, kernel_size=5, padding='same', activation='relu'),
 tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
 tf.keras.layers.Flatten(),
 tf.keras.layers.Dense(1024, activation='relu'),
 tf.keras.layers.Dropout(0.4),
 tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

def fake():
    import numpy as np
    return np.random.rand(200,28,28), np.random.rand(200,1)
train, label = fake()
model.fit(
    train,
    label,
    epochs=6,
)
