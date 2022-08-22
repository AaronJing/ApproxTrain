"""ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os


from tensorflow.keras import backend     
from tensorflow.python.keras import engine      
from tensorflow.keras import layers      
from tensorflow.keras import models      
from tensorflow.keras import utils as keras_utils 
import tensorflow as tf
import imagenet_utils
from imagenet_utils import decode_predictions
from imagenet_utils import _obtain_input_shape
import os
preprocess_input = imagenet_utils.preprocess_input

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
MOMENTUM=0.9
EPSILON=1e-5
WD=0.00005
KERASWRAPPERS=True
from python.keras.layers.am_convolutional import AMConv2D
"""
if KERASWRAPPERS:
else:
    import AMDNN._convam_grad
    convammodule =  tf.load_op_library('/scratch/ka88/jg7534/tmp/AMTensorflow/AMDNN/convam_gpu.so')
    class AMCONV(tf.keras.layers.Layer):
      def __init__(self, filters, kernel_size, padding='VALID', strides=1,name=None,**kwargs):
        super().__init__(name,**kwargs)
        self.stride = strides
        self.padding = padding.upper()
        self.kernel_size = kernel_size
        self.filters = filters
      def build(self, input_shape):
        self.w = tf.Variable(
          tf.keras.initializers.GlorotUniform()(shape=(self.kernel_size,self.kernel_size,input_shape[3],self.filters)), name=self.name+'w')
        self.b = tf.Variable(tf.zeros([self.filters]), name=self.name+'b')
      def call(self, inputs):
        y = convammodule.convam(inputs,self.w,strides=(1,self.stride,self.stride,1),padding=self.padding,name="AMCONV"+self.name)
        #y = tf.nn.conv2d(inputs,self.w,strides=(1,self.stride,self.stride,1),padding=self.padding,name="TFNNCONV"+self.name)
        return tf.nn.bias_add(y,self.b,name="biasadditon"+self.name)
"""
def identity_block(input_tensor, kernel_size, filters, stage, block, AM=False):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    layers.Conv2D = AMConv2D if AM else layers.Conv2D
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
        exit(0)
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = layers.Conv2D(filters1, (1, 1),
            kernel_regularizer=tf.keras.regularizers.l2(WD),name=conv_name_base + '2a')(input_tensor) if KERASWRAPPERS else \
            AMCONV(filters1,1,name=conv_name_base+'2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, momentum=MOMENTUM, epsilon=EPSILON ,name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',kernel_regularizer=tf.keras.regularizers.l2(WD), name=conv_name_base + '2b')(x) if KERASWRAPPERS else \
                      AMCONV(filters2,kernel_size,padding='SAME',name=conv_name_base+'2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, momentum=MOMENTUM, epsilon=EPSILON ,name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),kernel_regularizer=tf.keras.regularizers.l2(WD), name=conv_name_base + '2c')(x) if KERASWRAPPERS else \
                    AMCONV(filters3,1,name=conv_name_base+'2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, momentum=MOMENTUM, epsilon=EPSILON ,name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               AM=False
               ):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    layers.Conv2D = AMConv2D if AM else layers.Conv2D
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
        exit(0)
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,kernel_regularizer=tf.keras.regularizers.l2(WD),
                      name=conv_name_base + '2a')(input_tensor) if KERASWRAPPERS else AMCONV(filters1,1,strides=strides[0],name=conv_name_base+'2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, momentum=MOMENTUM, epsilon=EPSILON ,name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',kernel_regularizer=tf.keras.regularizers.l2(WD),
                      name=conv_name_base + '2b')(x) if KERASWRAPPERS else AMCONV(filters2,kernel_size,padding='SAME',name=conv_name_base+'2b')(x) 
    x = layers.BatchNormalization(axis=bn_axis,momentum=MOMENTUM, epsilon=EPSILON , name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),kernel_regularizer=tf.keras.regularizers.l2(WD), name=conv_name_base + '2c')(x) if KERASWRAPPERS else AMCONV(filters3,1,name=conv_name_base+'2c')(x)
    x = layers.BatchNormalization(axis=bn_axis,momentum=MOMENTUM, epsilon=EPSILON , name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1),kernel_regularizer=tf.keras.regularizers.l2(WD), strides=strides,
                             name=conv_name_base + '1')(input_tensor) if KERASWRAPPERS else AMCONV(filters3,1,strides=strides[0],name=conv_name_base+'1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, momentum=MOMENTUM, epsilon=EPSILON ,name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def conv_block_small(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               first=False,
               strides=(2, 2),
               AM=False
               ):
    layers.Conv2D = AMConv2D if AM else layers.Conv2D
    filters1, filters2 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
        exit(0)
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    shortcut = input_tensor
    if first:
        in_channel = input_tensor.get_shape().as_list()[-1]
        if in_channel == filters2:
            if strides[0] == 1:
                shortcut = tf.identity(input_tensor)
            else:
                shortcut = layers.MaxPooling2D(strides, strides)(x)
        else:
            shortcut = layers.Conv2D(filters2, 1, padding='same', name=conv_name_base + "sc")(input_tensor)

    x = layers.Conv2D(filters1, kernel_size, padding='same',kernel_regularizer=tf.keras.regularizers.l2(WD),
            name=conv_name_base + '2a')(input_tensor) if KERASWRAPPERS else AMCONV(filters1,kernel_size,padding='SAME',name=conv_name_base+'2b')(input_tensor) 
    x = layers.BatchNormalization(axis=bn_axis,momentum=MOMENTUM, epsilon=EPSILON , name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',kernel_regularizer=tf.keras.regularizers.l2(WD),
                      name=conv_name_base + '2b')(x) if KERASWRAPPERS else AMCONV(filters2,kernel_size,padding='SAME',name=conv_name_base+'2b')(x) 
    x = layers.BatchNormalization(axis=bn_axis,momentum=MOMENTUM, epsilon=EPSILON , name=bn_name_base + '2b')(x)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def ResNet50(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=10, AM=False):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=28,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
        print("channel last confirmed")
    else:
        bn_axis = 1
        exit(0)

#   x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
#   x = layers.Conv2D(64, (3, 3),
#                     strides=(2, 2),
#                     padding='valid',kernel_regularizer=tf.keras.regularizers.l2(WD),
#                     name='conv1')(x) if KERASWRAPPERS else AMCONV(64,7,strides=2,name='conv1')(x)
#   x = layers.BatchNormalization(axis=bn_axis, momentum=MOMENTUM, epsilon=EPSILON ,name='bn_conv1')(x)
#   x = layers.Activation('relu')(x)
#   x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv_block(img_input, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), AM=AM)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', AM=AM)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', AM=AM)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', AM=AM)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', AM=AM)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', AM=AM)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', AM=AM)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', AM=AM)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', AM=AM)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', AM=AM)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', AM=AM)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', AM=AM)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', AM=AM)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', AM=AM)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', AM=AM)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', AM=AM)

    if include_top:
#        x = layers.AveragePooling2D((3, 3), name='avg_pool')(x)
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = engine.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model
def ResNet18(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=10, AM=False):
    """
    cifar 10 only
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=28,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
        print("channel last confirmed")
    else:
        bn_axis = 1
        exit(0)
    print(AM)
    layers.Conv2D = AMConv2D if AM else layers.Conv2D
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',kernel_regularizer=tf.keras.regularizers.l2(WD),
                      name='conv1')(x) if KERASWRAPPERS else AMCONV(64,7,strides=2,name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, momentum=MOMENTUM, epsilon=EPSILON ,name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv_block_small(x, 3, [64, 64], stage = 2, block='a',strides=(1,1), AM=AM)
    x = conv_block_small(x, 3, [64, 64], stage = 2, block='b', AM=AM)

    x = conv_block_small(x, 3, [128, 128], stage = 3, block='a', first=True, AM=AM)
    x = conv_block_small(x, 3, [128, 128], stage = 3 , block='b', AM=AM)

    x = conv_block_small(x, 3, [256, 256], stage = 4, block='a', first=True, AM=AM)
    x = conv_block_small(x, 3, [256, 256], stage = 4 , block='b', AM=AM)

    x = conv_block_small(x, 3, [512, 512], stage = 5, block='a', first=True, AM=AM)
    x = conv_block_small(x, 3, [512, 512], stage = 5 , block='b', AM=AM)

    if include_top:
        x = layers.AveragePooling2D((7, 7), name='avg_pool')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = engine.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='resnet18')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model
def ResNet34(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=10, AM=False):
    """
    cifar 10 only
    """
    layers.Conv2D = AMConv2D if AM else layers.Conv2D
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=32,
                                      min_size=28,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
        print("channel last confirmed")
    else:
        bn_axis = 1
        exit(0)

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',kernel_regularizer=tf.keras.regularizers.l2(WD),
                      name='conv1')(x) if KERASWRAPPERS else AMCONV(64,7,strides=2,name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, momentum=MOMENTUM, epsilon=EPSILON ,name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block_small(x, 3, [64, 64], stage = 2, block='a',strides=(1,1), AM=AM)
    x = conv_block_small(x, 3, [64, 64], stage = 2, block='b', AM=AM)
    x = conv_block_small(x, 3, [64, 64], stage = 2, block='c', AM=AM)

    x = conv_block_small(x, 3, [128, 128], stage = 3, block='a', first=True, AM=AM)
    x = conv_block_small(x, 3, [128, 128], stage = 3, block='b', AM=AM)
    x = conv_block_small(x, 3, [128, 128], stage = 3, block='c', AM=AM)
    x = conv_block_small(x, 3, [128, 128], stage = 3, block='d', AM=AM)

    x = conv_block_small(x, 3, [256, 256], stage = 4, block='a', first=True, AM=AM)
    x = conv_block_small(x, 3, [256, 256], stage = 4, block='b', AM=AM)
    x = conv_block_small(x, 3, [256, 256], stage = 4, block='c', AM=AM)
    x = conv_block_small(x, 3, [256, 256], stage = 4, block='d', AM=AM)
    x = conv_block_small(x, 3, [256, 256], stage = 4, block='e', AM=AM)
    x = conv_block_small(x, 3, [256, 256], stage = 4, block='f', AM=AM)

    x = conv_block_small(x, 3, [512, 512], stage = 5, block='a', first=True, AM=AM)
    x = conv_block_small(x, 3, [512, 512], stage = 5, block='b', AM=AM)
    x = conv_block_small(x, 3, [512, 512], stage = 5, block='c', AM=AM)
    if include_top:
        x = layers.AveragePooling2D((7, 7), name='avg_pool')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = engine.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='resnet34')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model
def ResNet50ImageNet(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000, AM=False):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    layers.Conv2D = AMConv2D if AM else layers.Conv2D
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=197,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
        print("channel last confirmed")
    else:
        bn_axis = 1
        exit(0)

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',kernel_regularizer=tf.keras.regularizers.l2(WD),
                      name='conv1')(x) if KERASWRAPPERS else AMCONV(64,7,strides=2,name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, momentum=MOMENTUM, epsilon=EPSILON ,name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), AM=AM)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', AM=AM)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', AM=AM)

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', AM=AM)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', AM=AM)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', AM=AM)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d', AM=AM)

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', AM=AM)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', AM=AM)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', AM=AM)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', AM=AM)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', AM=AM)
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', AM=AM)

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', AM=AM)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', AM=AM)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', AM=AM)

    if include_top:
        x = layers.AveragePooling2D((7, 7), name='avg_pool')(x)
        x = layers.Flatten()(x)
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = engine.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='resnet50')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        else:
            weights_path = keras_utils.get_file(
                'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='a268eb855778b3df3c7506639542a6af')
        model.load_weights(weights_path)
        if backend.backend() == 'theano':
            keras_utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights)

    return model

from python.keras.layers.amdenselayer import denseam
def lenet5(AM):
    layers.Conv2D = AMConv2D if AM else layers.Conv2D
    dense = denseam if AM else layers.Dense
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1 )),
        layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2),padding='same'),
        layers.Conv2D(filters=32, kernel_size=5, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        tf.keras.layers.Flatten(),
        dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        dense(10, activation='softmax')
    ])
def lenet31(AM=False):
    dense = denseam if AM else layers.Dense
    return tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1 )),
        tf.keras.layers.Flatten(),
        dense(300, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        dense(10, activation='softmax')
    ])
