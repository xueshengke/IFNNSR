import sys
# caffe path
sys.path.append('/ext/xueshengke/caffe-1.0/python')
import caffe
# import numpy as np
from caffe import layers as L, params as P, to_proto

################################################################################
def weight(bottom, share=1):
    ## pointwise product
    top = L.ElementWiseProduct(bottom, bias_term=True, share=share,
                               weight_filler=dict(type='constant', value=0.1),
                               bias_filler=dict(type='constant', value=0))
    return top


################################################################################
def smooth(bottom, channel=4, group=4, kernel=5, dilate=1):
    model = L.Convolution(bottom, num_output=channel, group=group, kernel_size=kernel,
                        stride=1, pad=dilate*(kernel-1)/2, dilation=dilate, weight_filler=dict(type='msra'),
                        bias_term=True, bias_filler=dict(type='constant'),
                        param=dict(lr_mult=1))
    return model
