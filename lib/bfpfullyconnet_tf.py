# -*- coding: utf-8 -*-
"""BFPFullyConnet_tf.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1t8_y1MASs344iZj7w0ug-ipVCoPe5fD5
"""

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Utils import bfp_quantize, to_exponent_mantissa_width
import math
import time

# Tensorflow
import tensorflow as tf

def transform_fc_online(tensor, exponent, mantissa, chnl_group):
    # Offline means the shared exponent is fixed
    #      it is deternmined during the pre-inference
    # Quantize the activation tensor along channel dimension
    # Here we require the input tensor has the shape: [batch, channel]
    # opt_exp_list: the shared exponent list for offline quantization
    shp = tf.shape(tensor)
    #print ("shape1:", shp[1], " opt_exp_list:", len(opt_exp_list))

    if (chnl_group == -1):
        chnl_group = shp[1]
    number_of_blocks = math.ceil(shp[1]/chnl_group)

    if shp[1] % chnl_group == 0:
        # shp[1] is divisible by block size
        # Therefore just one tensor will be created
        tensor = bfp_quantize(tensor, exponent, mantissa, quant_dim=len(tf.shape(tensor))-1)
    else:
        raise ValueError("Channel is not divisible by channel group while bfp quantizeing the FC")

    return tensor

def transform_fc_offline(tensor, exponent, mantissa, opt_exp_list):
    # Offline means the shared exponent is fixed
    #      it is deternmined during the pre-inference
    # Quantize the activation tensor along channel dimension
    # Here we require the input tensor has the shape: [batch, channel]
    # opt_exp_list: the shared exponent list for offline quantization
    shp = tensor.shape
    #print ("shape1:", shp[1], " opt_exp_list:", len(opt_exp_list))
    number_of_blocks = len(opt_exp_list)
    block_size = (int)(shp[1]/len(opt_exp_list))
    opt_exp_list = tf.Tensor(opt_exp_list).cuda()
    #print ("shp:", shp)
    #print ("opt_exp_list:", len(opt_exp_list))
    if shp[1] % block_size == 0:
        # shp[1] is divisible by block size
        # Therefore just one tensor will be created
        tensor = tf.reshape(tensor, (shp[0], number_of_blocks, block_size))
        opt_exp_list = tf.expand_dims(opt_exp_list, 0) ##### Need Unit test
        tensor = to_exponent_mantissa_width(tensor, opt_exp_list, mantissa, quant_dim=len(tf.shape(tensor))-1)
        tensor = tf.reshape(tensor, shp)
    else:
        raise ValueError("Channel is not divisible by channel group while bfp quantizeing the FC")

    return tensor
