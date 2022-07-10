
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numbers
import os

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables as variables_lib
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_nn_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.deprecation import deprecated_argument_lookup

from tensorflow.python.util.tf_export import tf_export
import tensorflow as tf
convam_module = tf.load_op_library('./convam_gpu.so')

_CHANNELS_LAST_FORMATS = frozenset({
    "NWC", "NHC", "NHWC", "NWHC", "NDHWC", "NDWHC", "NHDWC", "NHWDC", "NWDHC",
    "NWHDC"
})

def _get_sequence(value, n, channel_index, name):
  """Formats a value input for gen_nn_ops."""
  # Performance is fast-pathed for common cases:
  # `None`, `list`, `tuple` and `int`.
  if value is None:
    return [1] * (n + 2)

  # Always convert `value` to a `list`.
  if isinstance(value, list):
    pass
  elif isinstance(value, tuple):
    value = list(value)
  elif isinstance(value, int):
    value = [value]
  elif not isinstance(value, collections_abc.Sized):
    value = [value]
  else:
    value = list(value)  # Try casting to a list.

  len_value = len(value)

  # Fully specified, including batch and channel dims.
  if len_value == n + 2:
    return value

  # Apply value to spatial dims only.
  if len_value == 1:
    value = value * n  # Broadcast to spatial dimensions.
  elif len_value != n:
    raise ValueError("{} should be of length 1, {} or {} but was {}".format(
        name, n, n + 2, len_value))

  # Add batch and channel dims (always 1).
  if channel_index == 1:
    return [1, 1] + value
  else:
    return [1] + value + [1]

def amconvolution_internal(
    input,  # pylint: disable=redefined-builtin
    filters,
    strides=None,
    padding="VALID",
    data_format=None,
    dilations=None,
    name=None,
    call_from_convolution=True,
    num_spatial_dims=None,
    mant_mul_lut=''
    ):
  """Internal function which performs rank agnostic convolution.

  Args:
    input: See `convolution`.
    filters: See `convolution`.
    strides: See `convolution`.
    padding: See `convolution`.
    data_format: See `convolution`.
    dilations: See `convolution`.
    name: See `convolution`.
    call_from_convolution: See `convolution`.
    num_spatial_dims: (Optional.).  It is a integer describing the
      rank of the spatial dimensions.  For `1-D`, `2-D` and `3-D` convolutions,
      the value of `num_spatial_dims` is `1`, `2`, and `3`, respectively.
      This argument is only required to disambiguate the rank of `batch_shape`
      when `filter_shape.ndims is None` and `len(batch_shape) > 1`.  For
      backwards compatibility, if `num_spatial_dims is None` and
     `filter_shape.ndims is None`, then `len(batch_shape)` is assumed to be
     `1` (i.e., the input is expected to be
     `[batch_size, num_channels] + input_spatial_shape`
     or `[batch_size] + input_spatial_shape + [num_channels]`.

  Returns:
    A tensor of shape and dtype matching that of `input`.

  Raises:
    ValueError: If input and filter both have unknown shapes, or if
      `num_spatial_dims` is provided and incompatible with the value
      estimated from `filters.shape`.
  """
  if (not isinstance(filters, variables_lib.Variable) and
      not tensor_util.is_tf_type(filters)):
    with ops.name_scope("convolution_internal", None, [filters, input]):
      filters = ops.convert_to_tensor(filters, name='filters')
  if (not isinstance(input, ops.Tensor) and not tensor_util.is_tf_type(input)):
    with ops.name_scope("convolution_internal", None, [filters, input]):
      input = ops.convert_to_tensor(input, name="input")

  filters_rank = filters.shape.rank
  inputs_rank = input.shape.rank
  if num_spatial_dims is None:
    if filters_rank:
      num_spatial_dims = filters_rank - 2
    elif inputs_rank:
      num_spatial_dims = inputs_rank - 2
    else:
      raise ValueError("rank of input or filter must be known")
  elif filters_rank and filters_rank - 2 != num_spatial_dims:
    raise ValueError(
        "inconsistent estimate of spatial dims ({}) vs. actual passed "
        "num_spatial_dims ({}).  n was estimated as len(filters.shape) - 2, "
        "but filters shape is: {}".format(filters_rank, num_spatial_dims,
                                          filters.shape))

  if inputs_rank:
    num_batch_dims = inputs_rank - num_spatial_dims - 1  # Channel dimension.
  else:
    num_batch_dims = 1  # By default, assume single batch dimension.

  if num_spatial_dims not in {1, 2, 3}:
    raise ValueError(
        "num_spatial_dims (input.shape.ndims - num_batch_dims - 1) must be one "
        "of 1, 2 or 3 but saw {}.  num_batch_dims: {}.".format(
            num_spatial_dims, num_batch_dims))

  if data_format is None or data_format in _CHANNELS_LAST_FORMATS:
    channel_index = num_batch_dims + num_spatial_dims
  else:
    channel_index = num_batch_dims

  if dilations is None:
    dilations = _get_sequence(dilations, num_spatial_dims, channel_index,
                              "dilations")
    is_dilated_conv = False
  else:
    dilations = _get_sequence(dilations, num_spatial_dims, channel_index,
                              "dilations")
    is_dilated_conv = any(i != 1 for i in dilations)

  strides = _get_sequence(strides, num_spatial_dims, channel_index, "strides")

  if name:
    default_name = None
  else:  # Most common case.
    default_name = "AMConv2D"


  with ops.name_scope(name, default_name, [input, filters]) as name:
    if not is_dilated_conv:
      return convam_module.convam(
          input,
          filters,
          strides,
          padding=padding,
          data_format=data_format,
          dilations=dilations,
          name=name,
          mant_mul_lut=mant_mul_lut
          )
    else:
        raise ValueError("Dilation is not supported in current implementation")

@tf_export("nn.amconvolution", v1=[])
@dispatch.add_dispatch_support
def amconvolution_v2(  # pylint: disable=missing-docstring
    input,  # pylint: disable=redefined-builtin
    filters,
    strides=None,
    padding="VALID",
    data_format=None,
    dilations=None,
    name=None,
    mant_mul_lut=''):
  return amconvolution_internal(
      input,  # pylint: disable=redefined-builtin
      filters,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilations=dilations,
      name=name,
      mant_mul_lut=mant_mul_lut)

@ops.RegisterGradient("Convam")
def _convam_grad_cc(op,grad):
  #data_type = op.get_attr("T")
  dilations = op.get_attr("dilations")
  strides = op.get_attr("strides")
  padding = op.get_attr("padding")
  data_format = op.get_attr("data_format")
  mant_mul_lut = op.get_attr("mant_mul_lut")
  # shape_0 input shape_1 filter
  shape_0 = array_ops.shape(op.inputs[0])
  shape_1 = array_ops.shape(op.inputs[1])
  return  [convam_module.convam_input_grad(shape_0,op.inputs[1],grad,
          dilations=dilations,
          strides=strides,
          padding=padding,
          data_format=data_format,
          mant_mul_lut=mant_mul_lut
          ),
          convam_module.convam_filter_grad(shape_1,op.inputs[0], grad,
          dilations=dilations,
          strides=strides,
          padding=padding,
          data_format=data_format,
          mant_mul_lut=mant_mul_lut
          )]
