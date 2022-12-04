from tensorflow.python.framework import ops
from tensorflow.python.eager import context
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops

import tensorflow as tf
gen_matmulam = tf.load_op_library('./matmulamAFM32.so')
_resource_variable_type = None
@tf_export("linalg.matmulam", "matmulam")
@dispatch.add_dispatch_support
def matmulam(a,
           b,
           transpose_a=False,
           transpose_b=False,
           adjoint_a=False,
           adjoint_b=False,
           a_is_sparse=False,
           b_is_sparse=False,
           mant_mul_lut='', 
           mul='',
           name=None):
    with ops.name_scope(name, "MatMulAM", [a, b]) as name:
        if adjoint_a or adjoint_b or a_is_sparse or b_is_sparse:
            raise ValueError("Matmulam unsupported args: adjoint_a, adjoint_b, a_is_sparse, b_is_sparse.")
        if transpose_a and adjoint_a:
          raise ValueError("Only one of transpose_a and adjoint_a can be True.")
        if transpose_b and adjoint_b:
          raise ValueError("Only one of transpose_b and adjoint_b can be True.")

        if context.executing_eagerly():
          if not isinstance(a, (ops.EagerTensor, _resource_variable_type)):
            a = ops.convert_to_tensor(a, name="a")
          if not isinstance(b, (ops.EagerTensor, _resource_variable_type)):
            b = ops.convert_to_tensor(b, dtype_hint=a.dtype.base_dtype, name="b")
        else:
          a = ops.convert_to_tensor(a, name="a")
          b = ops.convert_to_tensor(b, dtype_hint=a.dtype.base_dtype, name="b")

        # TODO(apassos) remove _shape_tuple here when it is not needed.
        a_shape = a._shape_tuple()  # pylint: disable=protected-access
        b_shape = b._shape_tuple()  # pylint: disable=protected-access

        output_may_have_non_empty_batch_shape = (
            (a_shape is None or len(a_shape) > 2) or
            (b_shape is None or len(b_shape) > 2))

        if (not a_is_sparse and
            not b_is_sparse) and output_may_have_non_empty_batch_shape and False:
          # BatchMatmul does not support transpose, so we conjugate the matrix and
          # use adjoint instead. Conj() is a noop for real matrices.
          if transpose_a:
            a = conj(a)
            adjoint_a = True
          if transpose_b:
            b = conj(b)
            adjoint_b = True
          return gen_math_ops.batch_mat_mul_v2(
              a, b, adj_x=adjoint_a, adj_y=adjoint_b, name=name)

        # Neither matmul nor sparse_matmul support adjoint, so we conjugate
        # the matrix and use transpose instead. Conj() is a noop for real
        # matrices.
        if adjoint_a:
          a = conj(a)
          transpose_a = True
        if adjoint_b:
          b = conj(b)
          transpose_b = True

        use_sparse_matmul = False
        if a_is_sparse or b_is_sparse:
          sparse_matmul_types = [dtypes.bfloat16, dtypes.float32]
          use_sparse_matmul = (
              a.dtype in sparse_matmul_types and b.dtype in sparse_matmul_types)
        if ((a.dtype == dtypes.bfloat16 or b.dtype == dtypes.bfloat16) and
            a.dtype != b.dtype):
          # matmul currently doesn't handle mixed-precision inputs.
          use_sparse_matmul = True
        if use_sparse_matmul:
            #ret = gen_matmulam.MatMulAM(
            #  a, b, transpose_a=transpose_a, transpose_b=transpose_b, name=name)
            return gen_matmulam.MatMulAM(
              a = (tf.linalg.matrix_transpose(a) if transpose_a else a), b=(tf.linalg.matrix_transpose(b) if transpose_b else b), name=name, mant_mul_lut = mant_mul_lut)
          # sparse_matmul always returns float32, even with
          # bfloat16 inputs. This prevents us from configuring bfloat16 training.
          # casting to bfloat16 also matches non-sparse matmul behavior better.
          # return ret
          #if a.dtype == dtypes.bfloat16 and b.dtype == dtypes.bfloat16:
            #ret = cast(ret, dtypes.bfloat16)
          #return ret
        else:
          return gen_matmulam.MatMulAM(
              a = (tf.linalg.matrix_transpose(a) if transpose_a else a), b=(tf.linalg.matrix_transpose(b) if transpose_b else b), name=name, mant_mul_lut = mant_mul_lut)
          #return gen_matmulam.MatMulAM(
          #    a=a, b=b, transpose_a=transpose_a, transpose_b=transpose_b, name=name)
def _MatMulGradAgainstFirstOnly(op, grad):
  """Gradient for MatMul, only for the first input."""
  t_a = op.get_attr("transpose_a")
  t_b = op.get_attr("transpose_b")
  mant_mul_lut = op.get_attr("mant_mul_lut")
  b = math_ops.conj(op.inputs[1])
  if not t_a and not t_b:
    #grad_a = gen_matmulam.MatMulAM(grad, b, transpose_b=True)
    grad_a = gen_matmulam.MatMulAM(a=grad, b=tf.linalg.matrix_transpose(b),mant_mul_lut=mant_mul_lut)
  elif not t_a and t_b:
    grad_a = gen_matmulam.MatMulAM(a=grad, b=b, mant_mul_lut=mant_mul_lut)
  elif t_a and not t_b:
    grad_a = gen_matmulam.MatMulAM(a=b, b=tf.linalg.matrix_transpose(grad), mant_mul_lut=mant_mul_lut)
  elif t_a and t_b:
    grad_a = gen_matmulam.MatMulAM(a=tf.linalg.matrix_transpose(b), b=tf.linalg.matrix_transpose(grad), mant_mul_lut=mant_mul_lut)
  return grad_a, None


def _MatMulGradAgainstSecondOnly(op, grad):
  """Gradient for MatMul, only for the second input."""
  t_a = op.get_attr("transpose_a")
  t_b = op.get_attr("transpose_b")
  mant_mul_lut = op.get_attr("mant_mul_lut")
  a = math_ops.conj(op.inputs[0])
  if not t_a and not t_b:
    grad_b = gen_matmulam.MatMulAM(a=tf.linalg.matrix_transpose(a), b=grad, mant_mul_lut=mant_mul_lut)
  elif not t_a and t_b:
    grad_b = gen_matmulam.MatMulAM(a=tf.linalg.matrix_transpose(grad), b=a, mant_mul_lut=mant_mul_lut)
  elif t_a and not t_b:
    grad_b = gen_matmulam.MatMulAM(a=a, b=grad, mant_mul_lut=mant_mul_lut)
  elif t_a and t_b:
    grad_b = gen_matmulam.MatMulAM(a=tf.linalg.matrix_transpose(grad), b=tf.linalg.matrix_transpose(a),mant_mul_lut=mant_mul_lut)
  return None, grad_b
@ops.RegisterGradient("MatMulAM")
def _MatMulGrad(op, grad):
  """Gradient for MatMul."""
  try:
    skip_input_indices = op.skip_input_indices
    if skip_input_indices is not None:
      if 1 in skip_input_indices:
        return _MatMulGradAgainstFirstOnly(op, grad)
      elif 0 in skip_input_indices:
        return _MatMulGradAgainstSecondOnly(op, grad)
  except AttributeError:
    # No gradient skipping, so do the full gradient computation
    pass

  t_a = op.get_attr("transpose_a")
  t_b = op.get_attr("transpose_b")
  a = math_ops.conj(op.inputs[0])
  b = math_ops.conj(op.inputs[1])
  mant_mul_lut = op.get_attr("mant_mul_lut")
  if not t_a and not t_b:
    grad_a = gen_matmulam.MatMulAM(a=grad, b=tf.linalg.matrix_transpose(b),mant_mul_lut=mant_mul_lut)
    grad_b = gen_matmulam.MatMulAM(a=tf.linalg.matrix_transpose(a), b=grad, mant_mul_lut=mant_mul_lut)
  elif not t_a and t_b:
    grad_a = gen_matmulam.MatMulAM(a=grad, b=b)
    grad_b = gen_matmulam.MatMulAM(a=tf.linalg.matrix_transpose(grad), b=a, mant_mul_lut=mant_mul_lut)
  elif t_a and not t_b:
    grad_a = gen_matmulam.MatMulAM(a=b, b=tf.linalg.matrix_transpose(grad), mant_mul_lut=mant_mul_lut)
    grad_b = gen_matmulam.MatMulAM(a=a, b=grad, mant_mul_lut=mant_mul_lut)
  elif t_a and t_b:
    grad_a = gen_matmulam.MatMulAM(a=tf.linalg.matrix_transpose(b), b=tf.linalg.matrix_transpose(grad), mant_mul_lut=mant_mul_lut)
    grad_b = gen_matmulam.MatMulAM(a=tf.linalg.matrix_transpose(grad), b=tf.linalg.matrix_transpose(a), mant_mul_lut=mant_mul_lut)
  shape_a_static = a.get_shape()
  shape_b_static = b.get_shape()
  output_may_have_non_empty_batch_shape = (
      (shape_a_static.rank is None or shape_a_static.rank > 2) or
      (shape_b_static.rank is None or shape_b_static.rank > 2))
  batch_shapes_match = (
      shape_a_static[:-2].is_fully_defined() and
      shape_b_static[:-2].is_fully_defined() and
      shape_a_static[:-2] == shape_b_static[:-2])
  if (not output_may_have_non_empty_batch_shape) or batch_shapes_match:
    return grad_a, grad_b

  sa = array_ops.shape(a)
  sb = array_ops.shape(b)
  ra, rb = gen_array_ops.broadcast_gradient_args(sa[:-2], sb[:-2])
  grad_a = array_ops.reshape(math_ops.reduce_sum(grad_a, ra), sa)
  grad_b = array_ops.reshape(math_ops.reduce_sum(grad_b, rb), sb)
  return grad_a, grad_b
