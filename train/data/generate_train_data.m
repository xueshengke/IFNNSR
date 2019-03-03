clear; close all;
addpath ../util;
addpath(genpath(cd))

%% settings
folder = {
%     '/ext/xueshengke/dataset/BSDS200';
%     '/ext/xueshengke/dataset/BSDS200-aug';
    '/ext/xueshengke/dataset/DIV2K_train_HR';
    '/ext/xueshengke/dataset/DIV2K_valid_HR';
};
savepath = '/ext/xueshengke/train_IFNNSR';
scale = 4;  % scale factor: 2, 3, 4
patch_size = [180, 240]; % cannot be too large
stride = patch_size;
batch_size = 64;
border = 2;

%% initialization
data = [];
label = [];

%% generate data
for t = 1 : length(folder)
    
file_bmp = dir(fullfile(folder{t},'*.bmp'));
file_png = dir(fullfile(folder{t},'*.png'));
filepaths = [file_bmp; file_png];
total_num = length(filepaths);

for i = 1 : total_num
    fprintf('%d / %d %s\n', i, total_num, filepaths(i).name);
    image = imread(fullfile(folder{t}, filepaths(i).name));
    image = im2double(image);
    image = imresize(image, 0.5, 'bicubic');
    if size(image, 3) > 1
        image = rgb2ycbcr(image);
        image = image(:, :, 1); % only use Y channel
    end
    image_gnd = modcrop(image, scale);
%     image_gnd = image;
    image_bic = imresize(imresize(image_gnd, 1/scale, 'bicubic'), scale, 'bicubic');
    image_res = image_gnd - image_bic;

    image_bic_batch = imtobatch(image_bic, patch_size);
    image_res_batch = imtobatch(image_res, patch_size);
%     image_gnd_batch = imtobatch(image_gnd, patch_size);
    
    image_bic_batch_ht = hartleyTrans(image_bic_batch, 't');
    image_res_batch_ht = hartleyTrans(image_res_batch, 't');
%     image_gnd_batch_ht = hartleyTrans(image_gnd_batch, 't');
    
    image_bic_batch_ht_quad = quarterSplit(image_bic_batch_ht, border);
    image_res_batch_ht_quad = quarterSplit(image_res_batch_ht, border);
%     image_gnd_batch_ht_quad = quarterSplit(image_gnd_batch_ht, border);
    
    data = cat(4, data, image_bic_batch_ht_quad);
    label = cat(4, label, image_res_batch_ht_quad);
%     label = cat(4, label, image_gnd_batch_ht_quad);
end

end
dim = size(data);
count = dim(4);
order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, :, order);
% data = reshape(data, [dim(1) dim(2) 1 dim(3)]);
% label = reshape(label, [dim(1) dim(2) 1 dim(3)]);

%% writing to HDF5
chunksz = batch_size;

created_flag = false;
totalct = 0;
savefile = [savepath, '_x' num2str(scale) '.h5'];

for batchno = 1:floor(count/chunksz)
    last_read = (batchno-1)*chunksz;
    batchdata = data(:, :, :, last_read+1:last_read+chunksz); 
    batchlabs = label(:, :, :, last_read+1:last_read+chunksz);
    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savefile, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savefile);