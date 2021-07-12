import tensorflow as tf
import numpy as np
import sys
import numpy
import _convam_grad
from tensorflow.contrib.layers import flatten
numpy.set_printoptions(threshold=sys.maxsize)
convam_module = tf.load_op_library('convam_gpu.so')
# Test different shape


def test_convam(sess,_x,_w,_strides=[1, 1, 1, 1]):
    with tf.device('/gpu:0'):
        x = tf.placeholder(tf.float32, shape = _x.shape)
        W = tf.placeholder(tf.float32, shape = _w.shape)
        # Test VALID padding
        wx_convam = convam_module.convam(x, W, strides=_strides, padding='VALID')
        grad_filter_am = tf.gradients(wx_convam, W)
        grad_input_am = tf.gradients(wx_convam,x)

        wx_conv = tf.nn.conv2d(x, W, strides=_strides, padding='VALID')
        grad_filter = tf.gradients(wx_conv, W)
        grad_input = tf.gradients(wx_conv,x)

        # Test SAME padding
        wx_convam_same = convam_module.convam(x, W, strides=_strides, padding='SAME')
        grad_filter_am_same = tf.gradients(wx_convam_same, W)
        grad_input_am_same = tf.gradients(wx_convam_same,x)

        wx_conv_same = tf.nn.conv2d(x, W, strides=_strides, padding='SAME')
        grad_filter_same = tf.gradients(wx_conv_same, W)
        grad_input_same = tf.gradients(wx_conv_same,x)

        #evaluate
        forward_am = sess.run(wx_convam,feed_dict = {x:_x,W:_w})
        forward_conv = sess.run(wx_conv,feed_dict = {x:_x,W:_w})

        backward_filter_am = np.asarray(sess.run(grad_filter_am,feed_dict = {x:_x,W:_w}))
        backward_filter_conv = np.asarray(sess.run(grad_filter,feed_dict = {x:_x,W:_w}))

        backward_input_am = np.asarray(sess.run(grad_input_am ,feed_dict = {x:_x,W:_w}))
        backward_input_conv = np.asarray(sess.run(grad_input,feed_dict = {x:_x,W:_w}))

        forward_am_same = np.asarray(sess.run(wx_convam_same,feed_dict = {x:_x,W:_w}))
        forward_conv_same = np.asarray(sess.run(wx_conv_same,feed_dict = {x:_x,W:_w}))

        backward_filter_am_same = np.asarray(sess.run(grad_filter_am_same,feed_dict = {x:_x,W:_w}))
        backward_filter_conv_same = np.asarray(sess.run(grad_filter_same,feed_dict = {x:_x,W:_w}))

        backward_input_am_same = np.asarray(sess.run(grad_input_am_same ,feed_dict = {x:_x,W:_w}))
        backward_input_conv_same = np.asarray(sess.run(grad_input_same,feed_dict = {x:_x,W:_w}))
        success = True

        abs_err_forward_valid = np.abs(forward_am-forward_conv)/forward_conv
        abs_err_forward_valid = np.asarray(abs_err_forward_valid)
        max_abs_err_forward_valid = np.max(abs_err_forward_valid)
        err_forward_valid = np.mean(np.abs(forward_am-forward_conv)/forward_conv) 
        if err_forward_valid > 1e-7:
            print("case start err_forward_valid")

            # print(forward_am)
            # print(forward_conv)
            print(err_forward_valid)
            print(max_abs_err_forward_valid)
            print( _x.shape)
            print( _w.shape)
            print("VALID")
            print("case end")
            print()
            success = False

        abs_err_forward_same = np.abs(forward_am_same-forward_conv_same)/forward_conv_same
        abs_err_forward_same = np.asarray(abs_err_forward_same)
        max_abs_err_forward_same = np.max(abs_err_forward_same)
        err_forward_same = np.mean(np.abs(forward_am_same-forward_conv_same)/forward_conv_same)
        if  err_forward_same > 1e-7:
            print("case start err_forward_same")
            # print(forward_am_same)
            # print(forward_conv_same)
            print(err_forward_same)
            print(max_abs_err_forward_same)
            print(_x.shape)
            print(_w.shape)
            print("SAME")
            print("case end")
            print()
            success = False
        abs_err_backward_filter_valid = np.abs(backward_filter_am-backward_filter_conv)/backward_filter_conv
        abs_err_backward_filter_valid = np.asarray(abs_err_backward_filter_valid)
        max_abs_err_backward_filter_valid = np.max(abs_err_backward_filter_valid)
        err_backward_filter_valid = np.mean(np.abs(backward_filter_am-backward_filter_conv)/backward_filter_conv)
        if err_backward_filter_valid >  1e-7:
            print("case start err_backward_filter_valid")
            # print(backward_filter_am)
            # print(backward_filter_conv)
            print(err_backward_filter_valid)
            print(max_abs_err_backward_filter_valid)
            print( _x.shape)
            print( _w.shape)
            print("VALID FILTER")
            print("case end")
            print()
            success = False
        abs_err_backward_input_valid = np.abs(backward_input_am-backward_input_conv)/backward_input_conv
        abs_err_backward_input_valid = np.asarray(abs_err_backward_input_valid)
        max_abs_err_backward_input_valid = np.max( abs_err_backward_input_valid)
        err_backward_input_valid = np.mean(np.abs(backward_input_am-backward_input_conv)/backward_input_conv)
        if err_backward_input_valid > 1e-7:
            print("case start err_backward_input_valid")
            # print(backward_input_am)
            # print(backward_input_conv)
            print(err_backward_input_valid)
            print(max_abs_err_backward_input_valid)
            print( _x.shape)
            print( _w.shape)
            print("VALID INPUT")
            print("case end")
            print()
            success = False
        abs_err_backward_filter_same = np.abs(backward_filter_am_same-backward_filter_conv_same)/backward_filter_conv_same
        abs_err_backward_filter_same = np.asarray( abs_err_backward_filter_same)
        max_abs_err_backward_filter_same = np.max( abs_err_backward_filter_same)
        err_backward_filter_same = np.mean(np.abs(backward_filter_am_same-backward_filter_conv_same)/backward_filter_conv_same)
        if err_backward_filter_same >  1e-7:
            print("case start err_backward_filter_same")
            # print(backward_filter_am_same)
            # print(backward_filter_conv_same)
            print(err_backward_filter_same)
            print(max_abs_err_backward_filter_same)
            print( _x.shape)
            print( _w.shape)
            print("SAME FILTER")
            print("case end")
            print()
            success = False
        abs_err_backward_input_same = np.abs(backward_input_am_same- backward_input_conv_same)/ backward_input_conv_same
        abs_err_backward_input_same = np.asarray(abs_err_backward_input_same)
        max_abs_err_backward_input_same = np.max(abs_err_backward_input_same)
        err_backward_input_same = np.mean(np.abs(backward_input_am_same- backward_input_conv_same)/ backward_input_conv_same)
        if err_backward_input_same > 1e-7:
            print("case start err_backward_input_same")
            # print(backward_input_am_same)
            # print(backward_input_conv_same)
            print(err_backward_input_same)
            print(max_abs_err_backward_input_same)
            print( _x.shape)
            print( _w.shape)
            print("SAME INPUT")
            print("case end")
            print()
            success = False


        return success

def get_random_np(x):
    return np.random.rand(x[0],x[1],x[2],x[3]),np.random.rand(x[4],x[5],x[6],x[7])
def get_random_int_np(x):
    return np.random.randint(100,size=x[0:4]),np.random.randint(100,size=x[4:8])
shape_dict = {
    0:(1,4,4,1,4,4,1,1),
    1:(1,4,4,1,3,3,1,1),
    2:(1,4,4,1,2,2,1,1),
    3:(1,4,4,1,1,1,1,1),
    4:(1,4,8,1,4,4,1,1),
    5:(1,4,8,1,3,3,1,1),
    6:(1,4,8,1,2,2,1,1),
    7:(1,4,8,1,1,1,1,1),
    8:(1,4,8,1,4,5,1,1),
    9:(1,4,8,1,3,4,1,1),
    10:(1,4,8,1,2,1,1,1),
    11:(1,4,8,1,1,1,1,1),
    12:(1,4,8,2,4,5,2,3),
    13:(1,4,8,2,3,4,2,3),
    14:(1,4,8,2,2,1,2,3),
    15:(1,4,8,2,1,1,2,3),
    12:(1,8,4,3,5,4,3,4),
    13:(1,8,4,3,3,4,3,4),
    14:(1,8,4,3,2,1,3,4),
    15:(1,8,4,3,1,1,3,4),
    16:(1,20,20,3,5,4,3,4),
    17:(1,20,20,3,3,4,3,4),
    18:(1,20,20,3,2,1,3,4),
    19:(1,20,20,3,1,1,3,4),
    20:(128,32,32,1,5,5,1,6),
    21:(128,14,14,6,5,5,6,16),
    22:(4,4,4,1,2,2,1,2),
    23:(4,14,14,1,5,5,1,16),
}

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(0,1):
    for shape in shape_dict.values():
        x,w = get_random_np(shape)
        result = test_convam(sess=sess,_x=x,_w=w)
        print("test with shape x: "+str(shape[0:4])+" w: "+str(shape[4:8]) + str(result))
