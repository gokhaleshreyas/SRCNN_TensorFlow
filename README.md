# SRCNN_TensorFlow

## About

This is the implementation of SRCNN Paper using TensorFlow.

## Requirements

TensorFlow 1.15.5 \
Pillow (PIL) \
Numpy \
h5py \
MATLAB or GNU Octave (For Data Preprocessing)

## Implementation Details

In this implementation, the SRCNN network can be trained on both 1 channel and 3 channels. The data pre-processing can be done using either MATLAB or GNU Octave (The code is provided for the same).

## Use

### Data Pre-processing

The MATLAB as well as GNU Octave Code is given for data pre-processing. The training data will be generated in hdf5 format.
As mentioned in the paper, the author's have used the ImageNet dataset. But here, the 91 image dataset is used (mentioned by the authors in the same paper).

Steps:
1. Download the 91 image dataset from the author's website (Link is in the reference section) and put them in the 'Dataset' directory.
2. Run the appropriate code for train data generation. For example, to create training data for y-channel using GNU Octave, run the 'data_gen_h5_octave_ychannel.m' file in GNU Octave to get the 'train_91_ychannels_octave.h5' file for training in the train directory.
3. Run the appropriate code for test data generation. For example, to create test data for y-channel, run the 'data_gen_octave_rgb2ycbcr.m' file to get the test data in test directory.

Note: Both the train file and test images are needed to be generated before initiating the training.

### Training

The training requires GPU. To train the network, follow the following steps:
1. Open terminal in the SRCNN_TensorFlow folder.
2. type 'python main.py' and select the appropriate arguments to begin the training.
3. Set the '--do_train' value to True to start training the network.


### testing

To test the trained network, ensure that the trained weights are in the model directory. Follow the same steps as done in training, except, set the '--do_test' value to True to test the network performance.


### Arguments (To be changed while training)
The arguments are: \
-h, --help            show this help message and exit \
--do_train:  To Start training, default value: False \
--do_test: To Start testing, default value: False \
--train_dir: Enter a different training directory than default \
--valid_dir: Enter a different validation directory than default \
--test_dir: Enter a different testing directory than default \
--model_dir: Enter a different model directory than default \
--result_dir: Enter a different result directory than default \
--scale: Enter a scale value among 2,3,4, default value: 3 \
--learning_rate: Enter learning rate, default value: 1e-4 \
--epochs: Enter the number of epochs, default value: 1000 \
--n_channels: Enter number of channels (1(y-channel) or 3(rgb or ycbcr)), default value: 1 \
--batch_size: Enter the batch size, default value: 128 \
--momentum: Enter a momentum value for SGD Optimizer, default value: 0.9 \
--colour_format: Enter the colour format among ych, ycbcr, rgb, default value: ych \
--network_filters: Enter the string of number of filters in the network (9-1-5, 9-3-5, 9-5-5), default value: '9-1-5' \
--prepare_data: Enter the string for data preparation (used in data pre-processing), default value: 'matlab'

## Work To  Be done

1. There are some issues regarding the '9-3-5' and '9-5-5' filters when cross validating and testing the results of the trained network.
2. For 'rgb' channels, the PSNR goes upto 14 dB and then drops down eventually (For the default and some changed values of hyperparameters). That is to be rectified.

## References

- [Official Website][1]
    - Reference to the original Matlab and Caffe code.

- [jinsuyoo/srcnn][2]
    - Reference to the referred repository.

[1]: http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
[2]: https://github.com/jinsuyoo/srcnn
