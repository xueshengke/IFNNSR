%% extract parameters from caffe model and save them as MAT files
close all; clear; clc;
setenv('LC_ALL','C')    % remove all local configurations
setenv('GLOG_minloglevel','2')  % remove any log when loading caffe modules
addpath '/ext/xueshengke/caffe-1.0/matlab'; % change to your caffe path

%% parameters, change settings if necessary
gpu_id = 0;
caffe.set_mode_cpu(); % for CPU
% caffe.set_mode_gpu(); % for GPU
% caffe.set_device(gpu_id);

model = '../test_x2.prototxt';
weights = '../model/iter_5e4_s2_d4_c5_k5.caffemodel';
save_file = '../FNNSR_x2_d4_c5_k5.mat';

net = caffe.Net(model, weights, 'test');

for ii = 1:20
  model.weight{ii} = net.layers(['conv',num2str(ii)]).params(1).get_data();
  model.bias{ii} = net.layers(['conv',num2str(ii)]).params(2).get_data();
end

save(save_file, 'model');
caffe.reset_all();