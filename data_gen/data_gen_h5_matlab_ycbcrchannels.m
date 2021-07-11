clc;
clear all;
close all;

image_size = 33;
label_size = 21;
scale = 3;
stride = 14;

dataset='91';
dataset_path = 'C:\Shreyas\Programming\Python\Neural_Networks\Data\Dataset\';
data_path = strcat(dataset_path, dataset);

save_data = 'train_91_ycbcrchannels_matlab.h5';
save_dataset_path_1 = 'C:\Shreyas\Programming\Python\Neural_Networks\SRCNN\Data';
save_dataset_path_2 = '\RGB2YCBCR_MATLAB\';
save_dataset_path = strcat(save_dataset_path_1, save_dataset_path_2);
save_path = strcat(save_dataset_path, save_data);
if ~exist(save_dataset_path)
    mkdir(save_dataset_path);
end
data = zeros(1, image_size, image_size, 1);
label = zeros(1, label_size, label_size, 1);
padding = abs(image_size - label_size)/2;
count = 0;

filepath = dir(fullfile(data_path, '*.bmp'));

for i = 1: length(filepath)
    img = imread(fullfile(data_path, filepath(i).name));
    img = rgb2ycbcr(img);
    img_y = im2double(img(:,:,1));
    img_cb = im2double(img(:,:,2));
    img_cr = im2double(img(:,:,3));
    [h,w] = size(img_y);
    img_y_label = img_y(1:h-mod(h,scale), 1:w-mod(w,scale));
    img_y_input = imresize(imresize(img_y_label, 1/scale, 'bicubic'), [h, w],'bicubic');
    img_cb_label = img_cb(1:h-mod(h,scale), 1:w-mod(w,scale));
    img_cb_input = imresize(imresize(img_cb_label, 1/scale, 'bicubic'), [h, w],'bicubic');
    img_cr_label = img_cr(1:h-mod(h,scale), 1:w-mod(w,scale));
    img_cr_input = imresize(imresize(img_cr_label, 1/scale, 'bicubic'), [h, w],'bicubic');
    [h1, w1] = size(img_y_label);
    
    for x = 1:stride:h1-image_size+1
        for y = 1:stride:w1-image_size+1
            
            subimg_y_input = img_y_input(x:x+image_size-1, y:y+image_size-1);
            subimg_y_label = img_y_label(x+padding:x+padding+label_size-1, y+padding:y+padding+label_size-1);
            subimg_cb_input = img_cb_input(x:x+image_size-1, y:y+image_size-1);
            subimg_cb_label = img_cb_label(x+padding:x+padding+label_size-1, y+padding:y+padding+label_size-1);
            subimg_cr_input = img_cr_input(x:x+image_size-1, y:y+image_size-1);
            subimg_cr_label = img_cr_label(x+padding:x+padding+label_size-1, y+padding:y+padding+label_size-1);
            
            count = count + 1;
            
            data(1,:,:,count) = subimg_y_input;
            label(1,:,:,count) = subimg_y_label;
            data(2,:,:,count) = subimg_cb_input;
            label(2,:,:,count) = subimg_cb_label;
            data(3,:,:,count) = subimg_cr_input;
            label(3,:,:,count) = subimg_cr_label;
        end
    end
end
order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, :, order);

h5create(save_path,'/data',[3 image_size image_size count]);
h5create(save_path,'/label',[3 label_size label_size count]);
h5write(save_path,'/data',data);
h5write(save_path,'/label',label);
h5disp(save_path);