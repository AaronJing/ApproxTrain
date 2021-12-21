import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
convam_grad_module = tf.load_op_library('/home/jing/AMDNN/convam_gpu.so')

@ops.RegisterGradient("Convam")
def _convam_grad_cc(op,grad):
  dilations = op.get_attr("dilations")
  strides = op.get_attr("strides")
  padding = op.get_attr("padding")
  data_format = op.get_attr("data_format")
  shape_0, shape_1 = array_ops.shape_n([op.inputs[0], op.inputs[1]])
  return  [convam_grad_module.convam_input_grad(shape_0,op.inputs[1],grad,
          dilations=dilations,
          strides=strides,
          padding=padding,
          data_format=data_format),convam_grad_module.convam_filter_grad(op.inputs[0],shape_1,grad,
          dilations=dilations,
          strides=strides,
          padding=padding,
          data_format=data_format)]
