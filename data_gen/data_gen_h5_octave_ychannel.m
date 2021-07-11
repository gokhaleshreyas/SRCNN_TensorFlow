clc;
clear all;
close all;

pkg load image

image_size = 33;
label_size = 21;
scale = 3;
stride = 14;

dataset='91';
dataset_path = 'C:\Shreyas\Programming\Python\Neural_Networks\Data\Dataset\';
data_path = strcat(dataset_path, dataset);

save_data = 'train_91_ychannels_octave.h5';
save_dataset_path_1 = 'C:\Shreyas\Programming\Python\Neural_Networks\SRCNN\Data';
save_dataset_path_2 = '\RGB2YCBCR_OCTAVE\';
save_dataset_path = strcat(save_dataset_path_1, save_dataset_path_2);
save_path = strcat(save_dataset_path, save_data);

data = zeros(1, image_size, image_size, 1);
label = zeros(1, label_size, label_size, 1);
padding = abs(image_size - label_size)/2;
count=0;

filepath = dir(fullfile(data_path, '*.bmp'));

for i = 1: length(filepath)
    img = imread(fullfile(data_path, filepath(i).name));
    if length(size(img))!=3
      img = cat(3, img, img, img);
    end
    img = rgb2ycbcr(img);
    img = im2double(img(:,:,1));
    [h,w] = size(img);
    img_label = img(1:h-mod(h,scale), 1:w-mod(w,scale));
    [h1, w1] = size(img_label);
    img_input = imresize(imresize(img_label, 1/scale, 'bicubic'), [h1, w1],'bicubic');
    
    for x = 1:stride:h1-image_size+1
        for y = 1:stride:w1-image_size+1
            
            subimg_input = img_input(x:x+image_size-1, y:y+image_size-1);
            subimg_label = img_label(x+padding:x+padding+label_size-1, y+padding:y+padding+label_size-1);
            
            count = count + 1;
            
            data(1,:,:,count) = subimg_input;
            label(1,:,:,count) = subimg_label;
        end
    end
end
order = randperm(count);
data = data(1, :, :, order);
label = label(1, :, :, order);

save('-hdf5', save_path, 'data', 'label');