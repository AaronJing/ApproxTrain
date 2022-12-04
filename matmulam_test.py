
import tensorflow as tf
from python.ops.math_ops import matmulam
#tf.debugging.set_log_device_placement(True)

#a = tf.constant([1.2, 2, 3, 4, 5, 6, 1.2, 2, 3, 4, 5, 6], shape=[2,3, 2])
#b = tf.constant([7.3, 8, 9, 10, 11, 12, 7.3, 8, 9, 10, 11, 12], shape=[2,3, 2])
#with tf.GradientTape(persistent=True) as g:
#    g.watch(a)
#    g.watch(b)
#    d = tf.linalg.matmul(a, b, True)
#    c = matmulam(a, b, True, mant_mul_lut="lut/MBM_7.bin")
#print(g.gradient(d,a))
#print(g.gradient(d,b))
#print(d)
#print(g.gradient(c,a))
#print(g.gradient(c,b))
#print(c)



a = tf.constant([1.2, 2, 3, 4, 5, 6, 1.2, 2, 3, 4, 5, 6], shape=[4, 3, 1])
#b = tf.constant([7.3, 8, 9, 10, 11, 12, 7.3, 8, 9, 10, 11, 12, 7.3, 8, 9, 10, 11, 12, 7.3, 8, 9, 10, 11, 12], shape=[4, 3, 2])
b = tf.constant([7.3, 8, 9, 10, 11, 12], shape=[1, 3, 2])
with tf.GradientTape(persistent=True) as g:
    g.watch(a)
    g.watch(b)
    d = tf.linalg.matmul(a, b, True)
    c = matmulam(a, b, True, mant_mul_lut="lut/MBM_7.bin", mul="AFM32")
print(g.gradient(d,a))
print(g.gradient(d,b))
print(d)
print(g.gradient(c,a))
print(g.gradient(c,b))
print(c)
