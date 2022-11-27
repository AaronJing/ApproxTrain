
import tensorflow as tf
from python.ops.math_ops import matmulam
tf.debugging.set_log_device_placement(True)

a = tf.constant([1.2, 2, 3, 4, 5, 6], shape=[3, 2])
b = tf.constant([7.3, 8, 9, 10, 11, 12], shape=[3, 2])
c = matmulam(a, b, True)
print(c)
d = tf.linalg.matmul(a, b, True)
print(d)
#print(matmulam)
