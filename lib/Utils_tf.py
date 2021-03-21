# import math
import time
import sys
import logging
import numpy as np
logger = logging.getLogger(__name__)

# Tensorflow
import tensorflow as tf

# writer = SummaryWriter("./tensorboard/weight_quant_comp")

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
    value_log = tf.math.log(absolute) / tf.math.log(2.0)
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
    shp = tf.shape(array)
    maxexp = tf.expand_dims(maxexp, quant_dim)
    # NOTE THAT THIS -2 IS BECAUSE OF THE LEADING 1 AND THE FACT THAT THIS SHOULD BE IN 2s COMPLEMENT
    # Make the exponent_needed has the same shape with array
    exponent_needed = (MANTISSA_WIDTH-maxexp-2)*tf.ones(shp)
    #print (exponent_needed)

    twos = 2.0 * tf.ones(shp)
    first_mant_w = tf.math.pow(twos, exponent_needed)
    array = array*first_mant_w
    #print (array)
    # Half LSB rounding:
    array = tf.cast(array, dtype = tf.float32)
    # print(array[0, :, 0, 0]) # Uncomment to print integer values
    array = tf.cast((array/first_mant_w), dtype = tf.int32)
    

    # Apply clamp
    max_clamp = ((1-(1/2)**(MANTISSA_WIDTH-2))/(1-(1/2))) * tf.math.pow(twos, maxexp)
    max_clamp = max_clamp * tf.ones(shp)
    max_clamp = tf.cast(max_clamp, dtype = tf.int32)
    #print ("clamped:", (array > max_clamp).sum(), "shape:", array.shape)
    array = tf.math.minimum(array, max_clamp)

    min_clamp = -max_clamp
    array = tf.math.maximum(array, min_clamp)
    array = tf.cast(array, dtype = tf.int8)
    
    return array
#
# def smooth_hist(array, eps=0.0001):
#     # This implementation is refer to the mxnet quantization document:
#     # https://github.com/apache/incubator-mxnet/blob/e17b7e2947b3848ee1b41f8ec8abafe0d1c319ad/python/mxnet/contrib/quantization.py#L241
#     #print ("before smooth", array)
#     # array is a tensor
#     is_zeros = tf.to_float(array == 0)
#     is_nonzeros = tf.to_float(array != 0)
#     n_zeros = tf.reduce_sum(is_zeros)
#     n_nonzeros = tf.size(array) - n_zeros
#     if (n_nonzeros.item() == 0):
#         raise ValueError("All the values are zeros, the array shape is:", array)
#     eps1 = eps * tf.gather(tf.to_float(n_zeros), 0) / tf.gather(tf.to_float(n_nonzero), 0) # tf.item
#     #print("eps1:", eps1)
#     array = tf.to_float(array)
#     array += eps * is_zeros + (-eps1) * is_nonzeros
#     assert tf.reduce_sum(array <= 0) == 0, "Some negtive values are generated during smoothing the histogram"
#     return array
#
# def find_exp_KL_act(array, MANTISSA_WIDTH, EXPONENT_WIDTH, group=1, eps=0.0001, bins_factor=3):
#     # Find the proper exponent value instead of max_exp by minimize the KL_divergence
#     #   num_bins is used to construct the histogram/distribution
#     #   eps is used to smooth the histogram/distribution
#     # Assuming the input has shape [batch, channel, height, width]
#
#     # Reshape to [batch, channel, height*width]
#     with tf.device('/device:gpu:0'):
#         array = array.cuda()
#         orig_shape = tf.shape(array)
#         group = orig_shape[1] if (group>orig_shape[1]) else group # group is whole channel when group is -1
#         number_of_blocks = math.ceil(orig_shape[1]/group)
#         opt_exp = tf.zeros((1)) #torch.empty
#         max_exp = tf.zeros((1))
#         if orig_shape[1] % group == 0:
#             # Find the max_exp
#             array = torch.reshape(array, (orig_shape[0], number_of_blocks, group*orig_shape[2]))
#             exp_array = find_exponent(array, EXPONENT_WIDTH)
#             max_exp = find_max_exponent(exp_array, quant_dim=len(array.shape)-1)
#             max_exp = find_max_exponent(max_exp, quant_dim=0)
#             opt_exp = tf.identity(max_exp)
#             # Unsqeeze for quantization use
#             us_max_exp = tf.expand_dims(max_exp, 0)
#             # Compute the histogram of original internal features
#             orig_hist = []
#             orig_max = []
#             orig_min = []
#             orig_num_bins = []
#             min_kl_div = []
#             #print("bins factor:", bins_factor)
#             for i in range(number_of_blocks):
#                 flat_array = torch.flatten(array[:, i, :]) #tf.layers
#                 float_max = tf.gather(tf.reduce_max(flat_array, 0)[0], 0)
#                 float_min = tf.gather(tf.reduce_min(flat_array, 0)[0], 0)
#                 #print ("orignal max", float_max, "original min", float_min)
#                 target_max_int = (int)(math.ceil(float_max))
#                 target_min_int = (int)(math.floor(float_min))
#                 # For mobilenet only
#                 target_diff = target_max_int - target_min_int
#                 if (target_diff < 6):
#                     interval = 0.02
#                     #num_bins = 1 + (int)((target_max_int - target_min_int)/interval) #8-2 for resnet, 8-5 for inceptionv4 8-5 for vgg16
#                 else:
#                     #interval = (float_max - float_min)/ 128
#                     interval = 0.035
#                     #num_bins = 1 + (int)((target_max_int - target_min_int)/(2/(2**(MANTISSA_WIDTH-bins_factor)))) #8-2 for resnet, 8-5 for inceptionv4 8-5 for vgg16
#                 #num_bins = 1 + (int)((target_max_int - target_min_int)/interval) #8-2 for resnet, 8-5 for inceptionv4 8-5 for vgg16
#                 #num_bins = 1 + (int)((target_max_int - target_min_int)/(2/(2**(MANTISSA_WIDTH-bins_factor)))) #8-2 for resnet, 8-5 for inceptionv4 8-5 for vgg16
#
#                 float_board = abs(float_max) if (abs(float_max) > abs(float_min)) else abs(float_min)
#                 #float_interval = (2*float_board)/64 # Indicate how accurate the distribution needs
#
#                 #float_interval = (2*float_board)/64 if (float_min<0) else (float_board)/64
#
#                 float_interval = (float_max-float_min)/128 # Indicate how accurate the distribution needs 70.62-128
#                 num_bins = 1 + (int)((target_max_int - target_min_int)/float_interval)
#
#                 #num_bins = 1 + (int)((target_max_int - target_min_int)/(2/(2**(MANTISSA_WIDTH-bins_factor)))) #8-2 for resnet, 8-5 for inceptionv4 8-5 for vgg16
#                 target_hist = tf.histogram_fixed_width(flat_array, [target_min_int, target_max_int], nbins=num_bins)
#
#                 #print ("flat array", flat_array.shape)
#                 # Smoth the target histogram
#                 target_hist = smooth_hist(target_hist, eps)
#                 # Nomalize the target histogram
#                 target_hist = target_hist/tf.math.reduce_sum(target_hist)
#                 # Add information into list
#                 orig_hist.append(target_hist)
#                 orig_max.append(target_max_int)
#                 orig_min.append(target_min_int)
#                 orig_num_bins.append(num_bins)
#                 min_kl_div.append(sys.float_info.max)
#
#             # Quantize accodingly, Here we only explore (max_exp-6) ~ max_exp
#             for i in range(3):
#                 quantized_array = to_exponent_mantissa_width(array, us_max_exp-i, MANTISSA_WIDTH,
#                                                             quant_dim=len(tf.shape(array)-1)
#                 for j in range(number_of_blocks):
#                     flat_qarray = torch.flatten(quantized_array[:, j, :])
#                     if (((tf.reduce_max(flat_qarray, 0))[0].item() < orig_min[j])):
#                         continue
#                     quantized_hist = tf.histogram_fixed_width(flat_qarray, [orig_min[j], orig_max[j]], nbins=orig_num_bins[j])
#                     # Smoth the quantized histogram
#                     quantized_hist = smooth_hist(quantized_hist, eps)
#                     # Log-Nomalize the quantized histogram
#                     quantized_hist = quantized_hist/tf.reduce_sum(quantized_hist)
#                     quantized_hist = tf.math.log(quantized_hist)
#                     # Calculate the KL-Divergence
#                     kl_div = F.kl_div(quantized_hist, orig_hist[j])
#                     if (min_kl_div[j] > tf.gather(kl_div, 0)):
#                         opt_exp[j] = (max_exp[j]-i)
#                         min_kl_div[j] = tf.gather(kl_div, 0)
#         else:
#             raise ValueError("Channel is not divisible by group  while determining the opt exponent list the separated activation")
#         num_nequal = tf.reduce_sum(max_exp != opt_exp)
#         logging.debug("After minimizing the KL divergence, %d / %d shared act exponents are improved" % (num_nequal.item(), opt_exp.numel()))
#         opt_exp = opt_exp.int().data.tolist()
#         opt_exp = np.repeat(opt_exp, group)
#         max_exp = max_exp.int().data.tolist()
#         max_exp = np.repeat(max_exp, group)
#         #print ("kl div:", min_kl_div)
#         return opt_exp, max_exp
