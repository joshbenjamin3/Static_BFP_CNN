import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import math
import time

# Tensorflow
import tensorflow as tf
from Utils import bfp_quantize ## need to change this

def transform_weight(tensor, exponent, mantissa, filter_group):
    # Quantize the weight tensor along filter dimension
    # Here we require the weight has the shape: [filter, channel, k, k]
    # filter_group : Inditate the number of filters in one group, where one group shared the same exponenet

    shp = tf.shape(tensor)
    number_of_blocks = math.ceil(shp[0]/filter_group)
    if shp[0] % filter_group == 0:
        # shp[1] is divisible by block size
        # Therefore just one tensor will be created
        tensor = tf.reshape(tensor, (number_of_blocks, filter_group*shp[1]*shp[2]*shp[3]))
        tensor = bfp_quantize(tensor, exponent, mantissa, quant_dim=len(tf.shape(tensor))-1)
        tensor = tf.reshape(tensor, (shp[0], shp[1], shp[2], shp[3]))
        return tensor

    else:
        # shp[0] is not divisible by channel group
        # Therefore two tensors will be created
        input('Filter is not divisible by filter group')

        if number_of_blocks == 1:
            # This means that the depth is less than the block size, so just one tensor will be created
            tensor = tf.reshape(tensor, (1, shp[0]*shp[1]*shp[2]*shp[3]))
            tensor = bfp_quantize(tensor, exponent, mantissa, quant_dim=len(tf.shape(tensor))-1)
            tensor = tf.reshape(tensor, (shp[0], shp[1], shp[2], shp[3]))
            return tensor
        else:
            # Separate two part, tensor1 contain (number_of_blocks-1), tensor2 contain the rest
            first_filter = ((number_of_blocks-1)*filter_group)
            tensor1 = tensor[0 : first_filter, :, :, :]
            t1_shp = tf.shape(tensor1)
            tensor2 = tensor[first_filter : shp[0], :, :, :]
            t2_shp = tf.shape(tensor2)

            # Perform quantization
            tensor1 = tf.reshape(tensor1, (number_of_blocks-1, filter_group*shp[1]*shp[2]*shp[3]))
            tensor2 = tf.reshape(tensor2, (1, (shp[0]-first_first_filter)*shp[1]*shp[2]*shp[3]))
            tensor1 = bfp_quantize(tensor1, exponent, mantissa, quant_dim=len(tf.shape(tensor))-1)
            tensor2 = bfp_quantize(tensor2, exponent, mantissa, quant_dim=len(tf.shape(tensor))-1)

            # Reshape and put back to original tensor
            tensor1 = tf.reshape(tensor1, t1_shp)
            tensor2 = tf.reshape(tensor2, t2_shp)
            tensor[0 : first_filter, :, :, :] = tensor1
            tensor[first_filter : shp[0], :, :, :] = tensor2
            return tensor

    return tensor
