import math
import time
import sys
import logging
import numpy as np
logger = logging.getLogger(__name__)

# Tensorflow
import tensorflow as tf

writer = SummaryWriter("./tensorboard/weight_quant_comp")

def bfp_quantize(tensor, EXPONENT_WIDTH, MANTISSA_WIDTH, quant_dim):
    # Quantize the tensor along quant_dim as Block Floating Point
    # For activation with shape [batch, channel, height, width]:
    #       quantized activation has shape [batch, num_channel_block, data]
    # For weight with shape []:
    #       quantized weight has shape [batch, num_filter_block, weight]
    v_exponent = find_exponent(tensor, EXPONENT_WIDTH)
    max_exponent = find_max_exponent(v_exponent, quant_dim)
    quantized_tensor = to_exponent_mantissa_width(tensor, max_exponent, MANTISSA_WIDTH, quant_dim)
    return quantized_tensor

def find_exponent(array, EXPONENT_WIDTH):
    # This receives an array of shape:
    # [number_of_blocks, channel, bs_size, h, w]
    MAX = 2**(EXPONENT_WIDTH-1)-1
    MIN = -2**(EXPONENT_WIDTH-1)
    absolute = tf.math.abs(array)
    value_log = np.log2(absolute)
    value_log = tf.clip_by_value(value_log, MIN, MAX)
    v_exponent = tf.math.floor(value_log)
    return v_exponent

def find_max_exponent(array, quant_dim):
    # Find the max exponent along the dim dimension
    # This receives an array of shape:
    # [number_of_blocks, channel, bs_size, h, w]
    max_exponent = tf.math.reduce_max(array, axis=quant_dim)
    # The return is of shape [number_of_blocks, channel, h, w]
    return max_exponent

def find_min_exponent(array, quant_dim):
    # Find the min exponent along the dim dimension
    # This receives an array of shape:
    # [number_of_blocks, channel, bs_size, h, w]
    min_exponent = tf.math.reduce_min(array, quant_dim)

    # The return is of shape [number_of_blocks, channel, h, w]
    return min_exponent

def to_exponent_mantissa_width(array, maxexp, MANTISSA_WIDTH, quant_dim):
    # This receives an array of shape:
    # [number_of_blocks, channel, bs_size, h, w]
    shp = array.shape
    maxexp = maxexp.tf.expand_dims(quant_dim)
    # NOTE THAT THIS -2 IS BECAUSE OF THE LEADING 1 AND THE FACT THAT THIS SHOULD BE IN 2s COMPLEMENT
    # Make the exponent_needed has the same shape with array
    exponent_needed = (MANTISSA_WIDTH-maxexp-2)*tf.ones(shp)
    #print (exponent_needed)
    first_mant_w = math.pow(2, exponent_needed)
    array = array*first_mant_w
    #print (array)
    # Half LSB rounding:
    array = tf.math.round(array)
    # print(array[0, :, 0, 0]) # Uncomment to print integer values
    array = array/first_mant_w

    # Apply clamp
    max_clamp = ((1-(1/2)**(MANTISSA_WIDTH-2))/(1-(1/2))) * math.pow(2, maxexp)
    max_clamp = max_clamp * tf.ones(shp)
    #print ("clamped:", (array > max_clamp).sum(), "shape:", array.shape)
    array = tf.math.minimum(array, max_clamp)

    min_clamp = -max_clamp
    array = tf.math.maximum(array, min_clamp)

    return array

def smooth_hist(array, eps=0.0001):
    # This implementation is refer to the mxnet quantization document:
    # https://github.com/apache/incubator-mxnet/blob/e17b7e2947b3848ee1b41f8ec8abafe0d1c319ad/python/mxnet/contrib/quantization.py#L241
    #print ("before smooth", array)
    # array is a tensor
    is_zeros = tf.to_float(array == 0)
    is_nonzeros = tf.to_float(array != 0)
    n_zeros = tf.reduce_sum(is_zeros)
    n_nonzeros = tf.size(array) - n_zeros
    if (n_nonzeros.item() == 0):
        raise ValueError("All the values are zeros, the array shape is:", array)
    eps1 = eps * tf.to_float(n_zeros).item() / tf.to_float(n_nonzero).item()
    #print("eps1:", eps1)
    array = tf.to_float(array)
    array += eps * is_zeros + (-eps1) * is_nonzeros
    assert tf.reduce_sum(array <= 0) == 0, "Some negtive values are generated during smoothing the histogram"
    return array
