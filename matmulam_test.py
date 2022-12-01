
import tensorflow as tf
#from python.ops.math_ops import matmulam
tf.debugging.set_log_device_placement(True)

a = tf.constant([1, 2, 3, 4, 5, 6], shape=[3, 1, 2])
b = tf.constant([7, 8, 9, 10], shape=[2, 2, 1])
#c = matmulam(a, b, True)
#print(c)
d = tf.linalg.matmul(a, b, True)
print(d)
#print(matmulam)
