import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Layer

#class PositionEmbeddingFixedWeights(Layer):
#    def __init__(self, seq_length, vocab_size, output_dim, **kwargs):
#        super().__init__(**kwargs)
#        word_embedding_matrix = self.get_position_encoding(vocab_size, output_dim)
#        pos_embedding_matrix = self.get_position_encoding(seq_length, output_dim)
#        self.word_embedding_layer = Embedding(
#            input_dim=vocab_size, output_dim=output_dim,
#            weights=[word_embedding_matrix],
#            trainable=False
#        )
#        self.position_embedding_layer = Embedding(
#            input_dim=seq_length, output_dim=output_dim,
#            weights=[pos_embedding_matrix],
#            trainable=False
#        )
#
#    def get_position_encoding(self, seq_len, d, n=10000):
#        P = np.zeros((seq_len, d))
#        for k in range(seq_len):
#            for i in np.arange(int(d/2)):
#                denominator = np.power(n, 2*i/d)
#                P[k, 2*i] = np.sin(k/denominator)
#                P[k, 2*i+1] = np.cos(k/denominator)
#        return P
#
#
#    def call(self, inputs):
#        position_indices = tf.range(tf.shape(inputs)[-1])
#        embedded_words = self.word_embedding_layer(inputs)
#        embedded_indices = self.position_embedding_layer(position_indices)
#        return embedded_words + embedded_indices
def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)

  return tf.cast(pos_encoding, dtype=tf.float32)

class PositionEmbeddingFixedWeights(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
    super().__init__()
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
    self.pos_encoding = positional_encoding(length=2048, depth=d_model)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x
