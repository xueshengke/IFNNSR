# This file creates IFNNSR prototxt files: 'train_val' for training and 'deploy' for test
from __future__ import print_function
import sys
# caffe path
sys.path.append('/ext/xueshengke/caffe-1.0/python')
sys.path.append('util')
import caffe
# import numpy as np
import function
from caffe import layers as L, params as P, to_proto
# from caffe.proto import caffe_pb2
from function import weight, smooth

################################################################################
# change filename here
train_net_path = 'train_x3.prototxt'
test_net_path = 'test_x3.prototxt'
train_data_path = 'examples/IFNNSR/train_data_x3.txt'
test_data_path = 'examples/IFNNSR/test_data_x3.txt'

# parameters of the network
batch_size_train = 64
batch_size_test = 2
scale = 4
depth = 5
channel = 16
kernel = 7
share = 4
group = 4 # 4
dilate = 1 # 1

################################################################################
# define the network for training and validation
def train_IFNNSR():
    net = caffe.NetSpec()
    net.data, net.label = L.HDF5Data(hdf5_data_param={'source': train_data_path,
        'batch_size': batch_size_train}, include={'phase': caffe.TRAIN}, ntop=2)
    train_data_layer = str(net.to_proto())
    net.data, net.label = L.HDF5Data(hdf5_data_param={'source': test_data_path,
        'batch_size': batch_size_test}, include={'phase': caffe.TEST}, ntop=2)

    net.model = net.data
    net.model = weight(net.model, share)    # element-wise product
    net.model = smooth(net.model, channel, group, kernel, dilate)   # convolution
    net.model = L.TanH(net.model)   # tanh actiavtion function
    net.model = smooth(net.model, channel, group, kernel, dilate)   # convolution
    net.model = L.TanH(net.model)   # tanh actiavtion function
    net.model = smooth(net.model, share, group, kernel, dilate)   # convolution
    net.model = L.TanH(net.model)   # tanh actiavtion function
    net.sum = net.model

    for j in range(depth-1):
        net.model = weight(net.model, share)
        net.model = smooth(net.model, channel, group, kernel, dilate)
        net.model = L.TanH(net.model)
        net.model = smooth(net.model, channel, group, kernel, dilate)
        net.model = L.TanH(net.model)
        net.model = smooth(net.model, share, group, kernel, dilate)
        net.model = L.TanH(net.model)
        net.sum = L.Eltwise(net.sum, net.model)

    net.predict = weight(net.sum, share)
    net.loss = L.WeightL2Loss(net.predict, net.label)
    # net.loss = L.EuclideanLoss(net.predict, net.label)

    return train_data_layer + str(net.to_proto())

################################################################################
# deploy the network for test; no data, label, loss layers
def test_IFNNSR():
    net = caffe.NetSpec()

    net.data = L.Input(shape=dict(dim=[1, 1, 2*depth+1, 2*depth+1]), ntop=1)

    net.model = net.data
    net.model = weight(net.model, share)    # element-wise product
    net.model = smooth(net.model, channel, group, kernel, dilate)   # convolution
    net.model = L.TanH(net.model)   # tanh actiavtion function
    net.model = smooth(net.model, channel, group, kernel, dilate)   # convolution
    net.model = L.TanH(net.model)   # tanh actiavtion function
    net.model = smooth(net.model, share, group, kernel, dilate)   # convolution
    net.model = L.TanH(net.model)   # tanh actiavtion function
    net.sum = net.model

    for j in range(depth-1):
        net.model = weight(net.model, share)
        net.model = smooth(net.model, channel, group, kernel, dilate)
        net.model = L.TanH(net.model)
        net.model = smooth(net.model, channel, group, kernel, dilate)
        net.model = L.TanH(net.model)
        net.model = smooth(net.model, share, group, kernel, dilate)
        net.model = L.TanH(net.model)
        net.sum = L.Eltwise(net.sum, net.model)

    net.predict = weight(net.sum, share)
    # net.loss = L.WeightL2Loss(net.predict, net.label)
    # net.loss = L.EuclideanLoss(net.predict, net.label)

    return net.to_proto()

################################################################################
if __name__ == '__main__':
    # write train_val network
    with open(train_net_path, 'w') as f:
        print(str(train_IFNNSR()), file=f)
    print('create ' + train_net_path)

    # write test network
    with open(test_net_path, 'w') as f:
        f.write('name: "IFNNSR'+'_scale='+str(scale)+'_depth='+str(depth)
                +'_channel='+str(channel)+'_kernel='+str(kernel)+'_share='+str(share)
                +'_group='+str(group)+'_dilate='+str(dilate)+'"\n')
        print(str(test_IFNNSR()), file=f)
    print('create ' + test_net_path)
