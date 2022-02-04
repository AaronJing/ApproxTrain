
import numpy as np
import sys
import _denseam_grad
import tensorflow as tf
# from tensorflow.contrib.layers import flatten

np.set_printoptions(threshold=sys.maxsize)
convam_module = tf.load_op_library('./denseam.so')

FLOAT = False

def float_comparison(x, y):
    return np.abs(x.numpy()-y.numpy())/y.numpy()

def int_comparison(x, y):
    return x.numpy().astype(int)-y.numpy().astype(int)

def test_convam(_x,_w):
    with tf.device('/gpu:0'):
        x = _x
        W = _w
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            g.watch(W)
            wx_denseam = convam_module.denseam(x, W)
            wx_dense = tf.matmul(x, W)
    
        grad_filter_am = g.gradient(wx_denseam, W)
        grad_input_am = g.gradient(wx_denseam,x)
        grad_filter = g.gradient(wx_dense, W)
        grad_input = g.gradient(wx_dense,x)

        #evaluate
        forward_am = wx_denseam
        forward_dense = wx_dense

        backward_filter_am = grad_filter_am
        backward_filter_dense = grad_filter

        backward_input_am =  grad_input_am
        backward_input_dense = grad_input

        success = True
        comparator = float_comparison if FLOAT else int_comparison
        abs_err_forward_valid = comparator(forward_am,forward_dense)
        abs_err_forward_valid = np.asarray(abs_err_forward_valid)
        max_abs_err_forward_valid = np.max(abs_err_forward_valid)
        err_forward_valid = np.mean(comparator(forward_am,forward_dense)) 
        if err_forward_valid > 1e-7:
            print("case start err_forward_valid")
            print(err_forward_valid)
            print(max_abs_err_forward_valid)
            print( _x.shape)
            print( _w.shape)
            print("VALID")
            print("case end")
            print()
            success = False


        abs_err_backward_filter_valid = comparator(backward_filter_am, backward_filter_dense)
        abs_err_backward_filter_valid = np.asarray(abs_err_backward_filter_valid)
        max_abs_err_backward_filter_valid = np.max(abs_err_backward_filter_valid)
        err_backward_filter_valid = np.mean(comparator(backward_filter_am, backward_filter_dense))
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
        abs_err_backward_input_valid = comparator(backward_input_am, backward_input_dense)
        abs_err_backward_input_valid = np.asarray(abs_err_backward_input_valid)
        max_abs_err_backward_input_valid = np.max( abs_err_backward_input_valid)
        err_backward_input_valid = np.mean(comparator(backward_input_am, backward_input_dense))
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

        return success

def get_random_np(x):
    return tf.convert_to_tensor(np.random.rand(x[0],x[1]),dtype=float),tf.convert_to_tensor(np.random.rand(x[1], x[2]),dtype=float)
def get_random_int_np(x):
    return tf.convert_to_tensor(np.random.randint(-2,2,size=x[0:2]),dtype=float),tf.convert_to_tensor(np.random.randint(-2,2,size=x[1:]),dtype=float)
shape_dict = {
        #batch input_width unit
        0:(2, 3, 4),
        1:(32, 784, 128),
        2:(32,128,10)
}


test_passed = True
for i in range(0, 4):
    for shape in shape_dict.values():
        x,w = get_random_np(shape) if FLOAT else get_random_int_np(shape) 
        result = test_convam(_x=x,_w=w)
        if result == False:
            test_passed = False
        print("test with shape x: "+str(shape[0:2])+" w: "+str(shape[1:]) + "Passed: "+ str(result))

exit(0) if test_passed else exit(1)
