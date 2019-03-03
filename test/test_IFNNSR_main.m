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

model = 'test_x4_d4_c16_k7_s4_g4_d1.prototxt';
weights = 'model/x4_d4_c16_k7_s4_g4_d1_iter_1e5.caffemodel';
data_set_id = 2;    % index to select one dataset for test
data_set = {
    'Set5',         '*.bmp';
    'Set14',        '*.bmp';
    'BSDS100',      '*.png';
    'Urban100',     '*.png';
};

scale = 4;  % scale factor: 2, 3, 4
patch_size = [180, 240]; % cannot be too large
stride = patch_size;
border = 2;         %  >=2 for noise

% choose dataset
path_folder = '/ext/xueshengke/dataset';
save_path = 'result';
test_set = data_set{data_set_id, 1};
path = fullfile(path_folder, test_set);
dic = dir(fullfile(path, data_set{data_set_id, 2}));

num_file = length(dic);
folder_result = fullfile(save_path, test_set, ['x', num2str(scale)]);

if ~exist(folder_result,'file')
    mkdir(folder_result);
end

%% process image files by IFNNSRNet
fnnsr_set = zeros(num_file, 3);
im_rec_set = cell(num_file, 1);
im_gnd_set = cell(num_file, 1);

for i = 1 : length(dic)   
    %% read image
    disp([num2str(i) '/' num2str(num_file) ', testing the image: ' dic(i).name]);
    file_name = dic(i).name;
    image_name = file_name(1:end-4);
    image  = imread(fullfile(path, file_name));
    image = im2double(image);
    
    %% work on Y channel only
    if size(image, 3) > 1
        image = rgb2ycbcr(image);
        image = image(:, :, 1);
    end
    image = modcrop(image, scale);
    image_gnd = image;
    [hei, wid, ch] = size(image_gnd); 
    image_bic = imresize(imresize(image_gnd, 1/scale, 'bicubic'), scale, 'bicubic');
    
    %% run pretrained network on subimages, to avoid blob size exceeding
    % remove border of each output subimage, to aovide the side effect of
    % convolution
    image_bic_batch = imtobatch(image_bic, patch_size);
    image_bic_batch_ht = hartleyTrans(image_bic_batch, 't');
    image_bic_batch_ht_quad = quarterSplit(image_bic_batch_ht, border);

    image_res_batch_ht_quad = zeros(size(image_bic_batch_ht_quad));
    for t = 1 : size(image_bic_batch_ht_quad, 4)
        image_res_batch_ht_quad(:, :, :, t) = IFNNSRNet(model, weights, ...
            image_bic_batch_ht_quad(:, :, :, t));
    end
    image_res_batch_ht = quarterMerge(image_res_batch_ht_quad, border);
    image_res_batch = hartleyTrans(image_res_batch_ht, 'i');
    image_res = batchtoim(image_res_batch, [hei, wid]);
    image_fnn = image_bic + image_res;
    
    %% visualize the images
    figure(i); 
    subplot(2,2,1); imshow(image_bic); title('input image');
    subplot(2,2,2); imshow(image_res); title('residual image');
    subplot(2,2,3); imshow(image_fnn); title('output image');
    subplot(2,2,4); imshow(image_gnd); title('ground truth');
   
    %% remove outside border
    image_fnn_s = shave(uint8(single(image_fnn) * 255), [border, border]);
    image_gnd_s = shave(uint8(single(image_gnd) * 255), [border, border]);
    im_rec_set{i} = image_fnn_s;
    im_gnd_set{i} = image_gnd_s;

    %% save image files
    imwrite(image_fnn_s, fullfile(folder_result, [image_name '_x' num2str(scale) '.png']));

    %% compute PSNR, RMSE, and SSIM
    fnnsr_set(i, 1) = compute_psnr(image_fnn_s, image_gnd_s);
    fnnsr_set(i, 2) = compute_rmse(image_fnn_s, image_gnd_s);
    fnnsr_set(i, 3) = ssim_index(image_fnn_s, image_gnd_s);
end

%% save PSNR, RMSE, and SSIM metrics
avg_fnnsr_res = mean(fnnsr_set);
psnr_set = fnnsr_set(:, 1);
rmse_set = fnnsr_set(:, 2);
ssim_set = fnnsr_set(:, 3);
save(fullfile(folder_result, [test_set '_PSNR_x' num2str(scale) '.mat']), 'psnr_set');
save(fullfile(folder_result, [test_set '_RMSE_x' num2str(scale) '.mat']), 'rmse_set');
save(fullfile(folder_result, [test_set '_SSIM_x' num2str(scale) '.mat']), 'ssim_set');

%% display results of IFNNSRNet
disp('------------- IFNNSRNet ------------');
disp('--- PSNR ------- RMSE ------ SSIM ---');
for i = 1 : length(psnr_set)
    fprintf('%10.4f    %8.4f    %8.4f\n', psnr_set(i), rmse_set(i), ssim_set(i));
end
disp('--- average of FNNSRNet = ');
fprintf('%10.4f    %8.4f    %8.4f\n', avg_fnnsr_res(1), avg_fnnsr_res(2), avg_fnnsr_res(3));
