from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.backend import softmax
from python.keras.layers.amdenselayer import denseam
from python.ops.math_ops import matmulam
import os
mul=os.environ['mul']
# Implementing the Scaled-Dot Product Attention
class DotProductAttention(Layer):
    def __init__(self, lut_file, **kwargs):
        super().__init__(**kwargs)
        self.lut_file = lut_file
    def call(self, queries, keys, values, d_k, mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        if mul == "FP32":
            scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))
        else:
            scores = matmulam(queries, keys, transpose_b=True, mant_mul_lut=self.lut_file) / math.sqrt(cast(d_k, float32))

        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask

        # Computing the weights by a softmax operation
        weights = softmax(scores)

        # Computing the attention by a weighted sum of the value vectors
        if mul == "FP32":
            return matmul(weights, values)
        return matmulam(weights, values, mant_mul_lut=self.lut_file)

# Implementing the Multi-Head Attention
class MultiHeadAttention(Layer):
    def __init__(self, lut_file, h, d_k, d_v, d_model, **kwargs):
        super().__init__(**kwargs)
        self.attention = DotProductAttention(lut_file)  # Scaled dot product attention
        self.heads = h  # Number of attention heads to use
        self.d_k = d_k  # Dimensionality of the linearly projected queries and keys
        self.d_v = d_v  # Dimensionality of the linearly projected values
        self.d_model = d_model  # Dimensionality of the model
        if mul == "FP32":
            self.W_q = Dense(d_k)   # Learned projection matrix for the queries
            self.W_k = Dense(d_k)   # Learned projection matrix for the keys
            self.W_v = Dense(d_v)   # Learned projection matrix for the values
            self.W_o = Dense(d_model) # Learned projection matrix for the multi-head output
        else:
            self.W_q = denseam(d_k, mant_mul_lut=lut_file)   # Learned projection matrix for the queries
            self.W_k = denseam(d_k, mant_mul_lut=lut_file)   # Learned projection matrix for the keys
            self.W_v = denseam(d_v, mant_mul_lut=lut_file)   # Learned projection matrix for the values
            self.W_o = denseam(d_model, mant_mul_lut=lut_file) # Learned projection matrix for the multi-head output

    def reshape_tensor(self, x, heads, flag):
        if flag:
            # Tensor shape after reshaping and transposing:
            # (batch_size, heads, seq_length, -1)
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], heads, -1))
            x = transpose(x, perm=(0, 2, 1, 3))
        else:
            # Reverting the reshaping and transposing operations:
            # (batch_size, seq_length, d_k)
            x = transpose(x, perm=(0, 2, 1, 3))
            x = reshape(x, shape=(shape(x)[0], shape(x)[1], self.d_k))
        return x

    def call(self, queries, keys, values, mask=None):
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.W_q(queries), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the keys to be able to compute all heads in parallel
        k_reshaped = self.reshape_tensor(self.W_k(keys), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange the values to be able to compute all heads in parallel
        v_reshaped = self.reshape_tensor(self.W_v(values), self.heads, True)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Compute the multi-head attention output using the reshaped queries,
        # keys, and values
        o_reshaped = self.attention(q_reshaped, k_reshaped, v_reshaped, self.d_k, mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

        # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.heads, False)
        # Resulting tensor shape: (batch_size, input_seq_length, d_v)

        # Apply one final linear projection to the output to generate the multi-head
        # attention. Resulting tensor shape: (batch_size, input_seq_length, d_model)
        return self.W_o(output)
