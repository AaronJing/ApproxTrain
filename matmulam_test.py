
import tensorflow as tf
from python.ops.math_ops import matmulam

a = tf.constant([1, 2, 3.4, 4, 5, 6], shape=[2, 3])
b = tf.constant([7, 8.3, 9, 10, 11, 12], shape=[3, 2])
c = matmulam(a, b)
print(c)
#print(matmulam)
