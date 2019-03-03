%% Code written by 
%% Shengke Xue, Ph.D., Zhejiang University, email: xueshengke@zju.edu.cn
%% test IFNNSRNet
close all; clear; clc;

setenv('LC_ALL','C')    % remove all local configurations
setenv('GLOG_minloglevel','2')  % remove any log when loading caffe modules
addpath '/ext/xueshengke/caffe-1.0/matlab'; % change to your caffe path
addpath(genpath('util'));

%% parameters, change settings if necessary
gpu_id = 0;
% caffe.set_mode_cpu(); % for CPU
caffe.set_mode_gpu(); % for GPU
caffe.set_device(gpu_id);

weights = 'model/simple_iter_200.caffemodel';
model = 'test_x2.prototxt';

input = ones(360, 480);
output = IFNNSRNet( model, weights, input );

diff = input * 1 - output;
sum(diff(:))

%% visualize the images
figure; 
subplot(1,2,1); imshow(input); title('input image');
subplot(1,2,2); imshow(output); title('output image');
