% load image package which is necessary for image processing functions

clc;
clear all;
close all;

base_path = '/Dataset';
folder = '91';
path = strcat(base_path, folder);

base_path = '/test/';

base_savepath_gt = '/RGB2YCBCR_MATLAB/91_gt_mat_rgb';
base_savepath_2x = '/RGB2YCBCR_MATLAB/91_2x_mat_rgb';
base_savepath_3x = '/RGB2YCBCR_MATLAB/91_3x_mat_rgb';
base_savepath_4x = '/RGB2YCBCR_MATLAB/91_4x_mat_rgb';

savepath_gt = strcat(base_path, base_savepath_gt);
savepath_2x = strcat(base_path, base_savepath_2x);
savepath_3x = strcat(base_path, base_savepath_3x);
savepath_4x = strcat(base_path, base_savepath_4x);

mkdir(savepath_gt);
mkdir(savepath_2x);
mkdir(savepath_3x);
mkdir(savepath_4x);

base_savepath_2x_upscaled = '/RGB2YCBCR_MATLAB/91_2x_upscaled_mat_rgb';
base_savepath_3x_upscaled = '/RGB2YCBCR_MATLAB/91_3x_upscaled_mat_rgb';
base_savepath_4x_upscaled = '/RGB2YCBCR_MATLAB/91_4x_upscaled_mat_rgb';

savepath_2x_upscaled = strcat(base_path, base_savepath_2x_upscaled);
savepath_3x_upscaled = strcat(base_path, base_savepath_3x_upscaled);
savepath_4x_upscaled = strcat(base_path, base_savepath_4x_upscaled);

mkdir(savepath_2x_upscaled);
mkdir(savepath_3x_upscaled);
mkdir(savepath_4x_upscaled);

filepath = dir(fullfile(path, '*.bmp'));

for i = 1:length(filepath)
    
    img = imread(fullfile(path, filepath(i).name));
    file_name = filepath(i).name;
    if length(size(img))<3
        img = cat(3,img, img, img); 
        %This is because, imread() returns a 2D matrix when a black and white or 
        %grayscale image and we need 3D matrix for rgb2ycbcr
        %img = ind2rgb(img, viridis);
    end
    [h,w,c] = size(img);
    img = img(1:h-mod(h,12), 1:w-mod(w,12), :);
    %img_ycbcr = rgb2ycbcr(img);
    img = im2double(img);
    
    img_clr = img(:, :, 2:3);
    img_y = img(:, :, 1);
    
    [h1, w1] = size(img_y);
    
    name_gt = sprintf('%s/%s', savepath_gt, file_name);
    imwrite(img, name_gt);
    
    img_clr_2x = imresize(img_clr, 1/2, 'bicubic');
    img_y_2x = imresize(img_y, 1/2, 'bicubic');
    name_2x = sprintf('%s/%s', savepath_2x, file_name);
    imwrite(cat(3, img_y_2x, img_clr_2x), name_2x);
    
    img_clr_3x = imresize(img_clr, 1/3, 'bicubic');
    img_y_3x = imresize(img_y, 1/3, 'bicubic');
    name_3x = sprintf('%s/%s', savepath_3x, file_name);
    imwrite(cat(3, img_y_3x, img_clr_3x), name_3x);
    
    img_clr_4x = imresize(img_clr, 1/4, 'bicubic');
    img_y_4x = imresize(img_y, 1/4, 'bicubic');
    name_4x = sprintf('%s/%s', savepath_4x, file_name);
    imwrite(cat(3, img_y_4x, img_clr_4x), name_4x);
    
    img_clr_2x_us = imresize(imresize(img_clr, 1/2, 'bicubic'), [h1, w1], 'bicubic');
    img_y_2x_us = imresize(imresize(img_y, 1/2, 'bicubic'), [h1, w1], 'bicubic');
    name_2x_us = sprintf('%s/%s', savepath_2x_upscaled, file_name);
    imwrite(cat(3, img_y_2x_us, img_clr_2x_us), name_2x_us);
    
    img_clr_3x_us = imresize(imresize(img_clr, 1/3, 'bicubic'), [h1, w1], 'bicubic');
    img_y_3x_us = imresize(imresize(img_y, 1/3, 'bicubic'), [h1, w1], 'bicubic');
    name_3x_us = sprintf('%s/%s', savepath_3x_upscaled, file_name);
    imwrite(cat(3, img_y_3x_us, img_clr_3x_us), name_3x_us);
    
    img_clr_4x_us = imresize(imresize(img_clr, 1/4, 'bicubic'), [h1, w1], 'bicubic');
    img_y_4x_us = imresize(imresize(img_y, 1/4, 'bicubic'), [h1, w1], 'bicubic');
    name_4x_us = sprintf('%s/%s', savepath_4x_upscaled, file_name);
    imwrite(cat(3, img_y_4x_us, img_clr_4x_us), name_4x_us);
    
end
