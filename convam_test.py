import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np
import sys
import numpy
import _convam_grad
tf.random.set_random_seed(2)
numpy.set_printoptions(threshold=sys.maxsize)
convam_module = tf.load_op_library('convam_gpu.so')

with tf.Session('') as sess:
  with tf.device('/gpu:0'):
    #output shape test
    # x = tf.constant([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],shape=(1,4,4,1),dtype='float')
    # #x = tf.constant([[[[1,2]],[[3,4]]]],dtype='float')
    # f = tf.constant([2,3,1,2,3,245,3,2,2,5,3,21,234,45,67,0],shape=(4,4,1,1),dtype='float')
    x = tf.placeholder(tf.float32, shape = (1,4,4,1))
    W = tf.placeholder(tf.float32, shape = (3,3,1,1))
    _W = np.asarray([1,2,3,4,5,6,7,8,9]).reshape((3,3,1,1))

    _x = np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1))
    
    convam = convam_module.convam(x, W, strides=[1, 2, 2, 1], padding='VALID')
    conv = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')
    convam_out = sess.run(convam,feed_dict={x:_x,W:_W})
    conv_out = sess.run(conv,feed_dict={x:_x,W:_W})

    print("Official")
    print(conv_out)
    print("AM: forward output")
    print(convam_out)
    # convam_shape = convam.shape
    grad_conv_input = sess.run(tf.gradients(conv,x),feed_dict={x:_x,W:_W})
    print("Official")
    print(grad_conv_input)
    grad_am_input = sess.run(tf.gradients(convam,x),feed_dict={x:_x,W:_W})
    print("AM: grad input")
    print(grad_am_input)
    abss = np.abs(np.asarray(grad_conv_input) - np.asarray(grad_am_input))/ np.asarray(grad_am_input)
    # grad_am_filter = sess.run(tf.gradients(convam,f))
    # print("AM: grad filter")
    # print(grad_am_filter)

  # conv = tf.nn.conv2d(x, f, strides=[1, 2, 2, 1], padding='VALID')
  # conv_out = conv.eval()
  # conv_shape = conv_out.shape
  # grad_input = tf.gradients(conv,x)
  # print("grad input")
  # print(grad_input)
  # grad_filter = tf.gradients(conv,f)
  # print("grad filter")
  # print(grad_filter)
  # print("input")
  # print(x.eval())
  # print("AM_OUTPUT")
  # print(convam_out)
  # print("AM_SHAPE")
  # print(convam_shape)
  # print("Acutual_OUTPUT")
  # print(conv_out)
  # print("Acutual_SHAPE")
  # print(conv_shape)
  # #output shape test

  """
  Evaluate the native conv2d gradient
  """
  # with tf.device('/cpu:0'):
  #   x = tf.placeholder(tf.float32, shape = (2,4,4,2))
  #   W = tf.constant([1],shape=(1,1,2,2),dtype='float')
  #   Wx_dense = convam_module.convam(x, W, strides=[1, 2, 2, 1], padding='SAME')
  #   # grad_x_dense = tf.gradients(Wx_dense, x)
  #   # grad_xx = tf.gradients(Wx_dense,W)
  #   # gradient_dense = sess.run( grad_x_dense, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float64)})
  #   result = sess.run( Wx_dense, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]).reshape((2,4,4,2)).astype(np.float32)})
  #   # grad = sess.run(grad_xx,feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float64)})
  #   # print(gradient_dense)
  #   # print(grad)
  #   print(result)
  # with tf.device('/gpu:0'):
  #   x = tf.placeholder(tf.float32, shape = (2,4,4,2))
  #   W = tf.constant([1],shape=(1,1,2,2),dtype='float')
  #   Wx_dense = convam_module.convam(x, W, strides=[1, 2, 2, 1], padding='SAME')
  #   # grad_x_dense = tf.gradients(Wx_dense, x)
  #   # grad_xx = tf.gradients(Wx_dense,W)
  #   # gradient_dense = sess.run( grad_x_dense, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float64)})
  #   result = sess.run( Wx_dense, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]).reshape((2,4,4,2)).astype(np.float32)})
  #   # grad = sess.run(grad_xx,feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float64)})
  #   # print(gradient_dense)
  #   # print(grad)
  #   print(result)


  # # x = tf.placeholder(tf.float32, shape = (1,4,4,1))
  # # W = tf.constant([1,2,3,4],shape=(2,2,1,1),dtype='float')
  # # Wx_dense = tf.nn.conv2d(x, W, strides=[1, 3, 3, 1], padding='VALID')
  # # grad_filter = tf.gradients(Wx_dense, W)
  # # grad_input = tf.gradients(Wx_dense,x)
  # # gradient_filter = sess.run(grad_filter, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float64)})
  # # final_results = sess.run(Wx_dense, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float64)})
  # # gradient_input = sess.run(grad_input, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float64)})
  # # print(gradient_filter)
  # # print(final_results)

  # #padding case
  # x = tf.placeholder(tf.float32, shape = (2,4,4,2))
  # W = tf.constant([1],shape=(1,1,2,2),dtype='float')
  # Wx_dense = tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')
  # grad_filter = tf.gradients(Wx_dense, W)
  # grad_input = tf.gradients(Wx_dense,x)
  # # gradient_filter = sess.run(grad_filter, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float64)})
  # final_results = sess.run(Wx_dense, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]).reshape((2,4,4,2)).astype(np.float32)})
  # # gradient_input = sess.run(grad_input, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float64)})
  # # print(gradient_input)
  # # print(gradient_filter)
  # print(final_results)




  # with tf.device('/cpu:0'):
  #   x = tf.placeholder(tf.float32, shape = (4,4,4,1))
  #   W = tf.constant([1,2,3,4,5,6,7,8],shape=(2,2,1,2),dtype='float')
  #   Wx_dense = convam_module.convam(x, W, strides=[1, 1, 1, 1], padding='VALID')
  #   grad_x_dense = tf.gradients(Wx_dense, x)
  #   grad_xx = tf.gradients(Wx_dense,W)
  #   # gradient_dense = sess.run( grad_x_dense, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float64)})
  #   result = sess.run( grad_x_dense, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]).reshape((4,4,4,1)).astype(np.float32)})
  #   #grad = sess.run(grad_xx,feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float64)})
  #   # print(gradient_dense)
  #   #print(grad)
  #   print(result)
  # with tf.device('/gpu:0'):
  #   x = tf.placeholder(tf.float32, shape = (4,4,4,1))
  #   W = tf.constant([1,2,3,4,5,6,7,8],shape=(2,2,1,2),dtype='float')
  #   Wx_dense = convam_module.convam(x, W, strides=[1, 1, 1, 1], padding='VALID')
  #   grad_x_dense = tf.gradients(Wx_dense, x)
  #   grad_xx = tf.gradients(Wx_dense,W)
  #   # gradient_dense = sess.run( grad_x_dense, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float64)})
  #   result = sess.run( grad_x_dense, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]).reshape((4,4,4,1)).astype(np.float32)})
  #   #grad = sess.run(grad_xx,feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float64)})
  #   # print(gradient_dense)
  #   #print(grad)
  #   print(result)


  # x = tf.placeholder(tf.float32, shape = (2,4,4,2))
  # W = tf.constant([1,2,3,4,5,6,7,8],shape=(2,2,2,1),dtype='float')
  # Wx_dense = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
  # grad_filter = tf.gradients(Wx_dense, W)
  # grad_input = tf.gradients(Wx_dense,x)
  # gradient_dense = sess.run( grad_filter, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]).reshape((2,4,4,2)).astype(np.float32)})
  # result = sess.run( Wx_dense, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]).reshape((2,4,4,2)).astype(np.float32)})
  # grad = sess.run(grad_input,feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]).reshape((2,4,4,2)).astype(np.float32)})
  # print(gradient_dense)



  # x = tf.placeholder(tf.float32, shape = (1,3,3,2))
  # W = tf.constant([1,2,3,4,5,6,7,8],shape=(2,2,2,1),dtype='float')
  # Wx_dense = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
  # grad_filter = tf.gradients(Wx_dense, W)
  # grad_input = tf.gradients(Wx_dense,x)
  # gradient_dense = sess.run( grad_filter, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]).reshape((1,3,3,2)).astype(np.float32)})
  
  # # result = sess.run( Wx_dense, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]).reshape((2,3,3,2)).astype(np.float32)})
  # # grad = sess.run(grad_input,feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]).reshape((2,3,3,2)).astype(np.float32)})
  # print(gradient_dense)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  #print(result)

  # #padding case
  # x = tf.placeholder(tf.float32, shape = (1,4,4,1))
  # W = tf.constant([1,2,3,4],shape=(2,2,1,1),dtype='float')
  # Wx_dense = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
  # grad_filter = tf.gradients(Wx_dense, W)
  # grad_input = tf.gradients(Wx_dense,x)
  # gradient_filter = sess.run(grad_filter, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float64)})
  # final_results = sess.run(Wx_dense, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float32)})
  # # gradient_input = sess.run(grad_input, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float64)})
  # # print(gradient_input)
  # print(gradient_filter)
  # #print(final_results)

  # with tf.device('/gpu:0'):
  #   x = tf.placeholder(tf.float32, shape = (1,32,32,1))
  #   W = tf.constant([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50],shape=(5,5,1,6),dtype='float')
  #   Wx_dense = convam_module.convam(x, W, strides=[1, 1, 1, 1], padding='SAME')
  #   grad_x_dense = tf.gradients(Wx_dense, x)
  #   # grad_xx = tf.gradients(Wx_dense,W)
  #   gradient_dense = sess.run( Wx_dense , feed_dict = {x: np.random.rand(1,32,32,1)})
  #   #result = sess.run( Wx_dense, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float32)})
  #   # grad = sess.run(grad_xx,feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float64)})
  #   print(gradient_dense)
  #   # print(grad)
  #   #print(result)
  # with tf.device('/gpu:0'):
  #   x = tf.placeholder(tf.float32, shape = (1,32,32,1))
  #   W = tf.constant([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50],shape=(5,5,1,6),dtype='float')

  #   Wx_dense = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
  #   grad_x_dense = tf.gradients(Wx_dense, x)
  #   # grad_xx = tf.gradients(Wx_dense,W)
  #   gradient_dense = sess.run( Wx_dense , feed_dict = {x: np.random.rand(1,32,32,1)})
  #   #result = sess.run( Wx_dense, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float32)})
  #   # grad = sess.run(grad_xx,feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]).reshape((1,4,4,1)).astype(np.float64)})
  #   print(gradient_dense)


  # with tf.device('/gpu:0'):
  #   x = tf.placeholder(tf.float32, shape = (1,3,3,2))
  #   W = tf.constant([1,2,3,4,5,6,7,8],shape=(2,2,2,1),dtype='float')
  #   Wx_dense = convam_module.convam(x, W, strides=[1, 1, 1, 1], padding='VALID')
  #   grad_x_dense = tf.gradients(Wx_dense, x)
  #   grad_xx = tf.gradients(Wx_dense,W)
  #   # gradient_dense = sess.run( grad_x_dense, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]).reshape((2,3,3,2)).astype(np.float32)})
  #   # result = sess.run( Wx_dense, feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36]).reshape((2,3,3,2)).astype(np.float32)})
  #   grad = sess.run(grad_xx,feed_dict = {x: np.asarray([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]).reshape((1,3,3,2)).astype(np.float32)})
  #   #print(gradient_dense)
  #   print(grad)
  #   #print(result)

  # with tf.device('/gpu:0'):
    
  #   x = tf.placeholder(tf.float32, shape = (10,4,4,2))
  #   # conv_w = tf.get_variable("conv",shape=[2,2,2,4],initializer=tf_contrib.layers.variance_scaling_initializer())
  #   conv_w = tf.constant([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],shape=(2,2,2,4),dtype='float')
  #   # print(conv_w.shape)
  #   conv_b = tf.Variable(tf.zeros(4))
  #   convam = convam_module.convam(x, conv_w, strides=[1, 1, 1, 1], padding='SAME')+conv_b

  #   conv_w1 = tf.constant([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64],shape=(2,2,4,4),dtype='float')
  #   conv_2d = tf.nn.conv2d(convam,conv_w1,strides=[1,1,1,1],padding='SAME')
  #   grad_x = tf.gradients(convam, x)
  #   grad_w = tf.gradients(conv_2d,conv_w)
  #   np = np.arange(0,320,1).reshape(10,4,4,2)
  #   tf.global_variables_initializer().run()
  #   # result = sess.run(convam,feed_dict = {x:np})
  #   grad_filter = sess.run(grad_w,feed_dict = {x:np})

