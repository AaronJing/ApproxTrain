import tensorflow as tf
import tensorflow.contrib as tf_contrib

import _convam_grad
#import _dense_grad
AM = True
convam_module = tf.load_op_library('./convam_gpu.so')
#dense_module = tf.load_op_library('cuda_op_kernel.so')

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf_contrib.layers.variance_scaling_initializer()
weight_regularizer = tf_contrib.layers.l2_regularizer(0.0001)


##################################################################################
# Layer
##################################################################################

def conv(x, channels, ag,kernel=4, stride=2, padding='SAME', use_bias=True, scope='conv_0'):
    with tf.variable_scope(scope):
 
        # x = tf.layers.conv2d(inputs=x, filters=channels,
        #                      kernel_size=kernel, kernel_initializer=weight_init,
        #                      kernel_regularizer=weight_regularizer,
        #                      strides=stride, use_bias=use_bias, padding=padding)
        # conv_b = tf.Variable(tf.zeros(channels))
        # conv_w = tf.get_variable("conv",shape=[kernel,kernel,x.get_shape().as_list()[3],channels],initializer=tf_contrib.layers.variance_scaling_initializer(),regularizer=weight_regularizer)

        
        # x = tf.nn.conv2d(input = x,filter=conv_w,strides=[1, stride, stride, 1],padding=padding)+conv_b


        if(AM):
            conv_w = tf.get_variable("conv",shape=[kernel,kernel,x.get_shape().as_list()[3],channels],initializer=tf_contrib.layers.variance_scaling_initializer(),regularizer=weight_regularizer)
            conv_b = tf.Variable(tf.zeros(channels))
            x = convam_module.convam(input = x, filter = conv_w, strides=[1, stride, stride, 1], padding=padding) + conv_b
        else:
            conv_b = tf.Variable(tf.zeros(channels))
            conv_w = tf.get_variable("conv",shape=[kernel,kernel,x.get_shape().as_list()[3],channels],initializer=tf_contrib.layers.variance_scaling_initializer(),regularizer=weight_regularizer)

            
            x = tf.nn.conv2d(input = x,filter=conv_w,strides=[1, stride, stride, 1],padding=padding)+conv_b
        

        return x

def fully_conneted(x, units, use_bias=True, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        # print(x.shape)
       # dense_w = tf.get_variable("dense",shape=[x.get_shape().as_list()[1],units],initializer=weight_init)
       # dense_b = tf.Variable(tf.zeros(units))
        # x = tf.matmul(x,dense_w)+dense_b
       # dense_b = tf.expand_dims(dense_b, 0)
        #x = dense_module.dense(x, dense_w, dense_b)
        return x

def resblock(x_init, channels, ag,is_training=True, use_bias=True, downsample=False,  scope='resblock') :
    with tf.variable_scope(scope) :

        x = batch_norm(x_init, is_training, scope='batch_norm_0')
        ag.append(x)
        x = relu(x)
        # 

        if downsample :
            x = conv(x, channels, ag,kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            x_init = conv(x_init, channels, ag,kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels,ag, kernel=3, stride=1, use_bias=use_bias, scope='conv_0')

        x = batch_norm(x, is_training, scope='batch_norm_1')
        ag.append(x)
        x = relu(x)
        
        x = conv(x, channels,ag, kernel=3, stride=1, use_bias=use_bias, scope='conv_1')



        return x + x_init

def bottle_resblock(x_init, channels, ag,is_training=True, use_bias=True, downsample=False, scope='bottle_resblock') :
    with tf.variable_scope(scope) :
        x = batch_norm(x_init, is_training, scope='batch_norm_1x1_front')
        ag.append(x)
        shortcut = relu(x)

        x = conv(shortcut, channels, ag,kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_front')
        x = batch_norm(x, is_training, scope='batch_norm_3x3')
        x = relu(x)

        if downsample :
            x = conv(x, channels, ag, kernel=3, stride=2, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels*4, ag, kernel=1, stride=2, use_bias=use_bias, scope='conv_init')

        else :
            x = conv(x, channels, ag ,kernel=3, stride=1, use_bias=use_bias, scope='conv_0')
            shortcut = conv(shortcut, channels * 4, ag,kernel=1, stride=1, use_bias=use_bias, scope='conv_init')

        x = batch_norm(x, is_training, scope='batch_norm_1x1_back')
        ag.append(x)
        x = relu(x)
        x = conv(x, channels*4, ag,kernel=1, stride=1, use_bias=use_bias, scope='conv_1x1_back')

        return x + shortcut



def get_residual_layer(res_n) :
    x = []

    if res_n == 18 :
        x = [2, 2, 2, 2]

    if res_n == 34 :
        x = [3, 4, 6, 3]

    if res_n == 50 :
        x = [3, 4, 6, 3]

    if res_n == 101 :
        x = [3, 4, 23, 3]

    if res_n == 152 :
        x = [3, 8, 36, 3]

    return x



##################################################################################
# Sampling
##################################################################################

def flatten(x) :
    return tf.layers.flatten(x)

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    return gap

def avg_pooling(x) :
    return tf.layers.average_pooling2d(x, pool_size=2, strides=2, padding='SAME')

##################################################################################
# Activation function
##################################################################################


def relu(x):
    return tf.nn.relu(x)


##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, is_training=True, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=None,
                                        is_training=is_training, scope=scope)

##################################################################################
# Loss function
##################################################################################

def classification_loss(logit, label) :
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=logit)
    loss = tf.reduce_mean(cross_entropy)
    prediction = tf.equal(tf.argmax(logit, -1), tf.argmax(label, -1))
    predictied_output = tf.argmax(logit,-1)
    actual_output = tf.argmax(label, -1)

    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return cross_entropy,loss, accuracy, predictied_output, actual_output



