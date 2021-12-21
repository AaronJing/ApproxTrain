
import numpy as np
import sys
import _convam_grad
import tensorflow as tf
# from tensorflow.contrib.layers import flatten

np.set_printoptions(threshold=sys.maxsize)
convam_module = tf.load_op_library('/home/jing/AMDNN/convam_gpu.so')

FLOAT = False

def float_comparison(x, y):
    return np.abs(x.numpy()-y.numpy())/y.numpy()

def int_comparison(x, y):
    return x.numpy().astype(int)-y.numpy().astype(int)

def test_convam(_x,_w,_strides=[1, 1, 1, 1]):
    with tf.device('/gpu:0'):
        x = _x
        W = _w
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            g.watch(W)
            wx_convam = convam_module.convam(x, W, strides=_strides, padding='VALID')
            wx_conv = tf.nn.conv2d(x, W, strides=_strides, padding='VALID')
            wx_convam_same = convam_module.convam(x, W, strides=_strides, padding='SAME')
            wx_conv_same = tf.nn.conv2d(x, W, strides=_strides, padding='SAME')
    
        grad_filter_am = g.gradient(wx_convam, W)
        grad_input_am = g.gradient(wx_convam,x)
        grad_filter = g.gradient(wx_conv, W)
        grad_input = g.gradient(wx_conv,x)
        grad_filter_am_same = g.gradient(wx_convam_same, W)
        grad_input_am_same = g.gradient(wx_convam_same,x)
        grad_filter_same = g.gradient(wx_conv_same, W)
        grad_input_same = g.gradient(wx_conv_same,x)

        #evaluate
        forward_am = wx_convam
        forward_conv = wx_conv

        backward_filter_am = grad_filter_am
        backward_filter_conv = grad_filter

        backward_input_am =  grad_input_am
        backward_input_conv = grad_input

        forward_am_same = wx_convam_same
        forward_conv_same = wx_convam_same

        backward_filter_am_same = grad_filter_am_same
        backward_filter_conv_same = grad_filter_same

        backward_input_am_same = grad_input_am_same
        backward_input_conv_same = grad_input_same
        success = True
        comparator = float_comparison if FLOAT else int_comparison
        abs_err_forward_valid = comparator(forward_am,forward_conv)
        abs_err_forward_valid = np.asarray(abs_err_forward_valid)
        max_abs_err_forward_valid = np.max(abs_err_forward_valid)
        err_forward_valid = np.mean(comparator(forward_am,forward_conv)) 
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

        abs_err_forward_same = comparator(forward_am_same,forward_conv_same)
        abs_err_forward_same = np.asarray(abs_err_forward_same)
        max_abs_err_forward_same = np.max(abs_err_forward_same)
        err_forward_same = np.mean(comparator(forward_am_same,forward_conv_same))
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

        abs_err_backward_filter_valid = comparator(backward_filter_am, backward_filter_conv)
        abs_err_backward_filter_valid = np.asarray(abs_err_backward_filter_valid)
        max_abs_err_backward_filter_valid = np.max(abs_err_backward_filter_valid)
        err_backward_filter_valid = np.mean(comparator(backward_filter_am, backward_filter_conv))
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
        abs_err_backward_input_valid = comparator(backward_input_am, backward_input_conv)
        abs_err_backward_input_valid = np.asarray(abs_err_backward_input_valid)
        max_abs_err_backward_input_valid = np.max( abs_err_backward_input_valid)
        err_backward_input_valid = np.mean(comparator(backward_input_am, backward_input_conv))
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
        abs_err_backward_filter_same = comparator(backward_filter_am_same,backward_filter_conv_same)
        abs_err_backward_filter_same = np.asarray( abs_err_backward_filter_same)
        max_abs_err_backward_filter_same = np.max( abs_err_backward_filter_same)
        err_backward_filter_same = np.mean(comparator(backward_filter_am_same,backward_filter_conv_same))
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
        abs_err_backward_input_same =comparator(backward_input_am_same, backward_input_conv_same)
        abs_err_backward_input_same = np.asarray(abs_err_backward_input_same)
        max_abs_err_backward_input_same = np.max(abs_err_backward_input_same)
        err_backward_input_same = np.mean(comparator(backward_input_am_same, backward_input_conv_same))
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
    return tf.convert_to_tensor(np.random.rand(x[0],x[1],x[2],x[3]),dtype=float),tf.convert_to_tensor(np.random.rand(x[4],x[5],x[6],x[7]),dtype=float)
def get_random_int_np(x):
    return tf.convert_to_tensor(np.random.randint(-2,2,size=x[0:4]),dtype=float),tf.convert_to_tensor(np.random.randint(-2,2,size=x[4:8]),dtype=float)
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
    16:(1,20,20,3,5,5,3,4),
    17:(1,20,20,3,5,4,3,4),
    18:(1,20,20,3,2,1,3,4),
    19:(1,20,20,3,1,1,3,4),
    20:(128,32,32,1,5,5,1,6),
    21:(128,14,14,6,5,5,6,16),
    22:(4,4,4,1,2,2,1,2),
    23:(4,14,14,1,5,5,1,16),
}


test_passed = True
for i in range(0,2):
    for shape in shape_dict.values():
        x,w = get_random_np(shape) if FLOAT else get_random_int_np(shape) 
        stride = [1,i+1,i+1,1]
        result = test_convam(_x=x,_w=w,_strides=stride)
        if result == False:
            test_passed = False
        print("test with shape x: "+str(shape[0:4])+" w: "+str(shape[4:8]) + "Stride: " + str(stride) + "Passed: "+ str(result))

exit(0) if test_passed else exit(1)
