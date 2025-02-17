# -*- coding: utf-8 -*-
"""BFPActivation_tf.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dVH36BLzbVnA3QqfCZYPNd3_RpsHU6vh
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Utils_tf import bfp_quantize, to_exponent_mantissa_width
import math
import time

# PyTorch
import torch

# Tensorflow
import tensorflow as tf

def transform_activation_online(tensor, exponent, mantissa, chnl_group, is_3d=False):
    # Online means the shared exponent is not fixed
    #      it is deternmined during the inference
    # Quantize the activation tensor along channel dimension
    # Here we require the input tensor has the shape: [batch, channel, heigh, widht]
    # chnl_group : Inditate the number of channel in one group, where one group shared the same exponenet
    if is_3d is True:
        orig_shape = tf.shape(tensor)
        tensor = tf.reshape(tensor, (orig_shape[0], orig_shape[1]*orig_shape[2], orig_shape[3], orig_shape[4]))
    shp = tf.shape(tensor)
    if (chnl_group == -1):
        chnl_group = shp[1]
    number_of_blocks = math.ceil(shp[1]/chnl_group)
    if shp[1] % chnl_group == 0:
        # shp[1] is divisible by block size
        # Therefore just one tensor will be created
        tensor = tf.reshape(tensor, (shp[0], number_of_blocks, chnl_group*shp[2]*shp[3]))
        tensor = bfp_quantize(tensor, exponent, mantissa, quant_dim=len(tf.shape(tensor))-1)
        tensor = tf.reshape(tensor, (shp[0], shp[1], shp[2], shp[3]))
        if is_3d is True:
            tensor = tf.reshape(tensor, (orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3], orig_shape[4]))
        return tensor

    else:
        # shp[1] is not divisible by channel group
        # Therefore two tensors will be created
        input('Channel is not divisible by channel group')

        if number_of_blocks == 1:
            # This means that the depth is less than the block size, so just one tensor will be created
            tensor = tf.reshape(tensor, (shp[0], 1, shp[1]*shp[2]*shp[3]))
            tensor = bfp_quantize(tensor, exponent, mantissa, quant_dim=len(tf.shape(tensor))-1)
            tensor = tf.reshape(tensor, (shp[0], shp[1], shp[2], shp[3]))
            return tensor
        else:
            # Separate two part, tensor1 contain (number_of_blocks-1), tensor2 contain the rest
            first_chnl = ((number_of_blocks-1)*chnl_group)
            tensor1 = tensor[:, 0 : first_chnl, :, :]
            t1_shp = tf.shape(tensor1)
            tensor2 = tensor[:, first_chnl : shp[1], :, :]
            t2_shp = tf.shape(tensor2)

            # Perform quantization
            tensor1 = tf.reshape(tensor1, (shp[0], number_of_blocks-1, chnl_group*shp[2]*shp[3]))
            tensor2 = tf.reshape(tensor2, (shp[0], 1, (shp[1]-first_chnl)*shp[2]*shp[3]))
            tensor1 = bfp_quantize(tensor1, exponent, mantissa, quant_dim=len(tensor1.shape)-1)
            tensor2 = bfp_quantize(tensor2, exponent, mantissa, quant_dim=len(tensor2.shape)-1)

            # Reshape and put back to original tensor
            tensor1 = tf.reshape(tensor1, t1_shp)
            tensor2 = tf.reshape(tensor2, t2_shp)
            tensor[:, 0 : first_chnl, :, :] = tensor1
            tensor[:, first_chnl : shp[1], :, :] = tensor2
            return tensor

    return tensor

def transform_activation_offline(tensor, exponent, mantissa, opt_exp_list, is_3d=False):
    # Offline means the shared exponent is fixed
    #      it is deternmined during the pre-inference
    # Quantize the activation tensor along channel dimension
    # Here we require the input tensor has the shape: [batch, channel, heigh, widht]
    # opt_exp_list: the shared exponent list for offline quantization
    if is_3d is True:
        orig_shape = tf.shape(tensor)
        tensor = tf.reshape(tensor, (orig_shape[0], orig_shape[1]*orig_shape[2], orig_shape[3], orig_shape[4]))
    shp = tensor.shape
    #print ("shape1:", shp[1], " opt_exp_list:", len(opt_exp_list))
    chnl_group = (int)(shp[1]/len(opt_exp_list))
    number_of_blocks = math.ceil(shp[1]/chnl_group)
    opt_exp_list = tf.Tensor(opt_exp_list).cuda()
    #print ("shap[1]:", shp)
    #print ("len exp list", len(opt_exp_list))
    if shp[1] % chnl_group == 0:
        # shp[1] is divisible by block size
        # Therefore just one tensor will be created
        #print (tensor.shape)
        tensor = tf.reshape(tensor, (shp[0], number_of_blocks, chnl_group*shp[2]*shp[3]))
        opt_exp_list = opt_exp_list.unsqueeze(0) ##### Need Unit test
        tensor = to_exponent_mantissa_width(tensor, opt_exp_list, mantissa, quant_dim=len(tf.shape(tensor))-1)
        tensor = tf.reshape(tensor, (shp[0], shp[1], shp[2], shp[3]))
        if is_3d is True:
            tensor = tf.reshape(tensor, (orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3], orig_shape[4]))
        return tensor

    else:
        # shp[1] is not divisible by channel group
        # Therefore two tensors will be created
        input('Channel is not divisible by channel group')

        if number_of_blocks == 1:
            # This means that the depth is less than the block size, so just one tensor will be created
            tensor = tf.reshape(tensor, (shp[0], 1, shp[1]*shp[2]*shp[3]))
            opt_exp_list = tf.expand_dims(opt_exp_list, 0) ##### Need Unit test
            tensor = to_exponent_mantissa_width(tensor, opt_exp_list, mantissa, quant_dim=len(tf.shape(tensor))-1)
            tensor = tf.reshape(tensor, (shp[0], shp[1], shp[2], shp[3]))
            return tensor
        else:
            # Separate two part, tensor1 contain (number_of_blocks-1), tensor2 contain the rest
            first_chnl = ((number_of_blocks-1)*chnl_group)
            tensor1 = tensor[:, 0 : first_chnl, :, :]
            t1_shp = tf.shape(tensor1)
            tensor2 = tensor[:, first_chnl : shp[1], :, :]
            t2_shp = tf.shape(tensor2)
            t1_exp_list = opt_exp_list[0:number_of_blocks-1]
            t1_exp_list = tf.expand_dims(t1_exp_list, 0)
            t2_exp_list = opt_exp_list[number_of_blocks-1]
            t2_exp_list = tf.expand_dims(t2_exp_list, 0)

            # Perform quantization
            tensor1 = tf.reshape(tensor1, (shp[0], number_of_blocks-1, chnl_group*shp[2]*shp[3]))
            tensor2 = tf.reshape(tensor2, (shp[0], 1, (shp[1]-first_chnl)*shp[2]*shp[3]))
            tensor1 = to_exponent_mantissa_width(tensor1, t1_exp_list, mantissa, quant_dim=len(tf.shape(tensor1))-1)
            tensor2 = to_exponent_mantissa_width(tensor2, t1_exp_list, mantissa, quant_dim=len(tf.shape(tensor2))-1)

            # Reshape and put back to original tensor
            tensor1 = tf.reshape(tensor1, t1_shp)
            tensor2 = tf.reshape(tensor2, t2_shp)
            tensor[:, 0 : first_chnl, :, :] = tensor1
            tensor[:, first_chnl : shp[1], :, :] = tensor2
            return tensor

    return tensor

def transform_activation_offline_3d(tensor, exponent, mantissa, opt_exp_list):
    # Offline means the shared exponent is fixed
    #      it is deternmined during the pre-inference
    # Quantize the activation tensor along channel dimension
    # Here we require the input tensor has the shape: [batch, channel, heigh, widht]
    # opt_exp_list: the shared exponent list for offline quantization
    shp = tf.shape(tensor)
    #print ("shape1:", shp[1], " opt_exp_list:", len(opt_exp_list))
    num_frame = shp[2]
    assert len(opt_exp_list) == num_frame
    chnl_group = (int)(shp[1]/len(opt_exp_list))
    number_of_blocks = math.ceil(shp[1]/chnl_group) * num_frame # use different exp for different frame
    opt_exp_list = tf.Tensor(opt_exp_list).cuda()
    if shp[1] % chnl_group == 0:
        # shp[1] is divisible by block size
        # Therefore just one tensor will be created
        tensor = tf.reshape(tensor, (shp[0], number_of_blocks, chnl_group*shp[2]*shp[3]))
        opt_exp_list = tf.expand_dims(opt_exp_list, 0) ##### Need Unit test
        tensor = to_exponent_mantissa_width(tensor, opt_exp_list, mantissa, quant_dim=len(tf.shape(tensor))-1)
        tensor = tf.reshape(tensor, (shp[0], shp[1], shp[2], shp[3], shp[4]))
        return tensor

    else:
        raise NotImplementedError

    return tensor

from google.colab import drive
drive.mount('/content/drive')
