
import tensorflow as tf
#from python.ops.math_ops import matmulam
tf.debugging.set_log_device_placement(True)

a = tf.constant([1, 2, 3, 4, 5, 6], shape=[3, 1, 2])
b = tf.constant([7, 8, 9, 10], shape=[2, 2, 1])
#c = matmulam(a, b, True)
#print(c)
a = tf.constant([1.2, 2, 3, 4, 5, 6], shape=[3, 2])
b = tf.constant([7.3, 8, 9, 10, 11, 12], shape=[3, 2])
d = tf.linalg.matmul(a, b, True)
print(d)
c = matmulam(a, b, True, mant_mul_lut="lut/ACC_7.bin")
print(c)
a = tf.constant([1.2, 2, 3, 4, 5, 6, 1.2, 2, 3, 4, 5, 6], shape=[4, 3, 1])
b = tf.constant([7.3, 8, 9, 10, 11, 12], shape=[1, 3, 2])
d = tf.linalg.matmul(a, b, True)
print(d)
c = matmulam(a, b, True, mant_mul_lut="lut/ACC_7.bin")
print(c)
#c = matmulam(a, b, True, mant_mul_lut="lut/MBM_7.bin")
#print(c)
#print(matmulam)
