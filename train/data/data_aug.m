function save_path = data_aug(data_path)
%% To do data augmentation
folder = data_path;
if folder(end) == '/'
    save_path = [folder(1:end-1) '-aug'];
else
    save_path = [folder '-aug'];
end

if ~exist(save_path, 'dir')
    mkdir(save_path);
end

file_bmp = dir(fullfile(folder,'*.bmp'));
file_png = dir(fullfile(folder,'*.png'));
filepaths = [file_bmp; file_png];

for i = 1 : length(filepaths)
    filename = filepaths(i).name;
    [add, im_name, type] = fileparts(filepaths(i).name);
    image = imread(fullfile(folder, filename));
    
    for angle = 0 : 1 : 3
        im_rot = rot90(image, angle); % 90 degree counterclockwise rotation
        imwrite(im_rot, [save_path '/' im_name, '_rot' num2str(angle*90) '.bmp']);
        
        for scale = 0.5 : 0.2 :0.9
            im_down = imresize(im_rot, scale, 'bicubic');
            imwrite(im_down, [save_path '/' im_name, '_rot' num2str(angle*90) '_s' num2str(scale*10) '.bmp']);
        end
        
    end
end
