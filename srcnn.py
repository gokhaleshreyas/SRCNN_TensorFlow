import tensorflow as tf
import numpy as np
import os
import h5py
import time
from PIL import Image


class SRCNN:
    def __init__(self, args, sess):
        self.sess = sess
        self.do_train = args.do_train
        self.do_test = args.do_test
        self.train_dir = args.train_dir
        self.test_dir = args.test_dir
        self.valid_dir = args.valid_dir
        self.model_dir = args.model_dir
        self.result_dir = args.result_dir
        self.scale = args.scale
        self.learning_rate = args.learning_rate
        self.epochs = args.epochs
        self.n_channels = args.n_channels
        self.batch_size = args.batch_size
        self.momentum = args.momentum
        self.colour_format = args.colour_format
        self.network_filters = args.network_filters
        self.prepare_data = args.prepare_data
        if self.colour_format == 'ych':
            self.model_name = 'srcnn_ych'
        elif self.colour_format == 'ycbcr':
            self.model_name = 'srcnn_ycbcr'
        elif self.colour_format == 'rgb':
            self.model_name = 'srcnn_rgb'
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.n_channels])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.n_channels])
        if self.network_filters == '9-1-5':
            self.weights = {
                'w1': tf.Variable(initial_value=tf.random_normal(shape=[9, 9, self.n_channels, 64], stddev=1e-3),
                                  dtype=tf.float32),
                'w2': tf.Variable(initial_value=tf.random_normal(shape=[1, 1, 64, 32], stddev=1e-3),
                                  dtype=tf.float32),
                'w3': tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 32, self.n_channels], stddev=1e-3),
                                  dtype=tf.float32)
            }
        elif self.network_filters == '9-3-5':
            self.weights = {
                'w1': tf.Variable(initial_value=tf.random_normal(shape=[9, 9, self.n_channels, 64], stddev=1e-3),
                                  dtype=tf.float32),
                'w2': tf.Variable(initial_value=tf.random_normal(shape=[3, 3, 64, 32], stddev=1e-3),
                                  dtype=tf.float32),
                'w3': tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 32, self.n_channels], stddev=1e-3),
                                  dtype=tf.float32)
            }
        elif self.network_filters == '9-5-5':
            self.weights = {
                'w1': tf.Variable(initial_value=tf.random_normal(shape=[9, 9, self.n_channels, 64], stddev=1e-3),
                                  dtype=tf.float32),
                'w2': tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 64, 32], stddev=1e-3),
                                  dtype=tf.float32),
                'w3': tf.Variable(initial_value=tf.random_normal(shape=[5, 5, 32, self.n_channels], stddev=1e-3),
                                  dtype=tf.float32)
            }
        self.biases = {
            'b1': tf.Variable(initial_value=tf.zeros(shape=[64], dtype=tf.float32)),
            'b2': tf.Variable(initial_value=tf.zeros(shape=[32], dtype=tf.float32)),
            'b3': tf.Variable(initial_value=tf.zeros(shape=[self.n_channels], dtype=tf.float32))
        }
        self.output = self.model()
        self.loss = tf.reduce_mean(tf.squared_difference(self.output, self.y))
        self.result = tf.clip_by_value(self.output, clip_value_min=0., clip_value_max=1.)
        self.saver = tf.train.Saver()
        self.optimizer = self.optimize()

    def optimize(self):
        var_list_1 = [self.weights['w1'], self.biases['b1'], self.weights['w2'], self.biases['b2']]
        var_list_2 = [self.weights['w3'], self.biases['b3']]
        optimizer_1 = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate, momentum=self.momentum
        ).minimize(loss=self.loss, var_list=var_list_1)
        optimizer_2 = tf.train.MomentumOptimizer(
            learning_rate=self.learning_rate, momentum=self.momentum
        ).minimize(loss=self.loss, var_list=var_list_2)
        optimizer = tf.group(optimizer_1, optimizer_2)
        return optimizer

    def model(self):
        conv1 = tf.nn.conv2d(input=self.X, filters=self.weights['w1'], strides=[1, 1, 1, 1],
                             padding='VALID')
        conv1 = tf.nn.bias_add(conv1, self.biases['b1'])
        activ1 = tf.nn.relu(conv1)
        conv2 = tf.nn.conv2d(input=activ1, filters=self.weights['w2'], strides=[1, 1, 1, 1],
                             padding='VALID')
        conv2 = tf.nn.bias_add(conv2, self.biases['b2'])
        activ2 = tf.nn.relu(conv2)
        conv3 = tf.nn.conv2d(input=activ2, filters=self.weights['w3'], strides=[1, 1, 1, 1],
                             padding='VALID')
        conv3 = tf.nn.bias_add(conv3, self.biases['b3'])
        return conv3

    def train(self):
        print("Training Will Start Shortly")
        if self.prepare_data == 'matlab':
            train_X, train_y = load_matlab_data(self.train_dir, self.colour_format)
            valid_X_bc, valid_y_bc = make_matlab_bc_data(self.valid_dir, self.scale, self.colour_format)
            valid_y_gt = make_matlab_gt_data(self.valid_dir, self.colour_format)
        elif self.prepare_data == 'octave':
            train_X, train_y = load_octave_data(self.train_dir, self.colour_format)
            valid_X_bc, valid_y_bc = make_octave_bc_data(self.valid_dir, self.scale, self.colour_format)
            valid_y_gt = make_octave_gt_data(self.valid_dir, self.colour_format)
        else:
            print("Invalid arguments for prepare_data")
        start_time = time.time()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.model()
        self.optimize()
        total_batches = len(train_X) // self.batch_size
        batch_size = self.batch_size
        for i in range(self.epochs):
            loss = 0
            a = start_time
            b = time.time()
            for j in range(total_batches):
                batch_X = train_X[j * batch_size:(j + 1) * batch_size]
                batch_y = train_y[j * batch_size:(j + 1) * batch_size]
                _, batch_error = self.sess.run([self.optimizer, self.loss],
                                               feed_dict={self.X: batch_X, self.y: batch_y})
                loss = loss + batch_error

            valid_psnr = []
            bicubic_psnr = []
            for k in range(len(valid_X_bc)):
                h1, w1, c1 = valid_X_bc[k].shape
                h2, w2, c2 = valid_y_bc[k].shape
                h3, w3, c3 = valid_y_gt[k].shape
                if self.n_channels == 1:
                    valid_X_bc_ych = valid_X_bc[k][:, :, 0]
                    valid_y_bc_ych = valid_y_bc[k][:, :, 0]
                    valid_y_gt_ych = valid_y_gt[k][:, :, 0]
                    valid_X_bc_ych = valid_X_bc_ych.reshape([1, h1, w1, 1])
                    valid_y_bc_ych = valid_y_bc_ych.reshape([1, h2, w2, 1])
                    valid_y_gt_ych = valid_y_gt_ych.reshape([1, h3, w3, 1])
                    results = self.sess.run(self.result, feed_dict={self.X: valid_X_bc_ych, self.y: valid_y_gt_ych})
                    valid_y_bc_ych = border_crop(valid_y_bc_ych[0])
                    valid_y_gt_ych = border_crop(valid_y_gt_ych[0])
                    results = results[0]
                    bicubic_psnr.append(psnr(valid_y_bc_ych, valid_y_gt_ych))
                    valid_psnr.append(psnr(results, valid_y_gt_ych))
                elif self.n_channels == 3:
                    valid_X_bc_ = valid_X_bc[k]
                    valid_y_bc_ = valid_y_bc[k]
                    valid_y_gt_ = valid_y_gt[k]
                    valid_X_bc_ = valid_X_bc_.reshape([1, h1, w1, c1])
                    valid_y_bc_ = valid_y_bc_.reshape([1, h2, w2, c2])
                    valid_y_gt_ = valid_y_gt_.reshape([1, h3, w3, c3])
                    results = self.sess.run(self.result, feed_dict={self.X: valid_X_bc_, self.y: valid_y_gt_})
                    valid_y_bc_ = border_crop(valid_y_bc_[0])
                    valid_y_gt_ = border_crop(valid_y_gt_[0])
                    results = results[0]
                    bicubic_psnr.append(psnr(valid_y_bc_, valid_y_gt_))
                    valid_psnr.append(psnr(results, valid_y_gt_))
                else:
                    print("Invalid Argument for n_channels")
            print(f"Epoch: {i + 1}, Bicubic PSNR: {np.mean(bicubic_psnr)}, SRCNN PSNR: {np.mean(valid_psnr)}"
                  f"Time: {b - a}")
        self.save()
        print("Training Complete")
        end_time = time.time()
        print("Time Taken: {}".format(end_time - start_time))

    def test(self):
        print("Testing will commence")
        if self.prepare_data == 'matlab':
            test_X_bc, test_y_bc = make_matlab_bc_data(self.test_dir, self.scale, self.colour_format)
            test_y_gt = make_matlab_gt_data(self.test_dir, self.colour_format)
        elif self.prepare_data == 'octave':
            test_X_bc, test_y_bc = make_octave_bc_data(self.test_dir, self.scale, self.colour_format)
            test_y_gt = make_octave_gt_data(self.test_dir, self.colour_format)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.load()
        bicubic_psnr = []
        test_psnr = []
        start_time = time.time()
        for i in range(len(test_X_bc)):
            h1, w1, c1 = test_X_bc[i].shape
            h2, w2, c2 = test_y_bc[i].shape
            h3, w3, c3 = test_y_gt[i].shape
            if self.n_channels == 1:
                test_X_bc_ych = test_X_bc[i][:, :, 0]
                test_y_bc_ych = test_y_bc[i][:, :, 0]
                test_y_gt_ych = test_y_gt[i][:, :, 0]
                test_y_bc_cbcr = test_y_bc[i][:, :, 1:3]
                test_X_bc_ych = test_X_bc_ych.reshape([1, h1, w1, 1])
                test_y_bc_ych = test_y_bc_ych.reshape([1, h2, w2, 1])
                test_y_gt_ych = test_y_gt_ych.reshape([1, h3, w3, 1])
                results = self.sess.run(self.result, feed_dict={self.X: test_X_bc_ych, self.y: test_y_gt_ych})
                test_y_bc_ych = border_crop(test_y_bc_ych[0])
                test_y_gt_ych = border_crop(test_y_gt_ych[0])
                results = results[0]
                gt = test_y_gt[i]
                bc = test_y_bc[i]
                srcnn = np.concatenate((results, test_y_bc_cbcr), axis=2)
                save_res(gt, bc, srcnn, i, self.colour_format)
                bicubic_psnr.append(psnr(test_y_bc_ych, test_y_gt_ych))
                test_psnr.append(psnr(results, test_y_gt_ych))
            elif self.n_channels == 3:
                test_X_bc_ = test_X_bc[i]
                test_y_bc_ = test_y_bc[i]
                test_y_gt_ = test_y_gt[i]
                test_X_bc_ = test_X_bc_.reshape([1, h1, w1, c1])
                test_y_bc_ = test_y_bc_.reshape([1, h2, w2, c2])
                test_y_gt_ = test_y_gt_.reshape([1, h3, w3, c3])
                results = self.sess.run(self.result, feed_dict={self.X: test_X_bc_, self.y: test_y_gt_})
                test_y_bc_ = border_crop(test_y_bc_[0])
                test_y_gt_ = border_crop(test_y_gt_[0])
                results = results[0]
                gt = test_y_gt[i]
                bc = test_y_bc[i]
                srcnn = results
                save_res(self.result_dir, gt, bc, srcnn, i, self.colour_format)

                bicubic_psnr.append(psnr(test_y_bc_, test_y_gt_))
                test_psnr.append(psnr(results, test_y_gt_))
            else:
                print("Invalid Argument for n_channels")

        for p in range(len(bicubic_psnr)):
            print("Bicubic PSNR of Image {}: {:.2f}".format(p, bicubic_psnr[p]))
        for q in range(len(test_psnr)):
            print("SRCNN PSNR of Image {}: {:.2f}".format(q, test_psnr[q]))
        print("Average Bicubic PSNR: {:.2f}".format(np.mean(bicubic_psnr)))
        print("Average SRCNN PSNR: {:.2f}".format(np.mean(test_psnr)))
        end_time = time.time()
        print("Time taken: {}".format(end_time - start_time))

    def save(self):
        path = self.model_dir
        if not os.path.exists(path):
            os.mkdir(self.model_dir)
        self.saver.save(self.sess, self.model_dir + self.model_name, global_step=self.epochs)

    def load(self):
        path = self.model_dir
        if path:
            checkpoint_path = tf.train.latest_checkpoint(path)
            self.saver.restore(self.sess, checkpoint_path)
            print("Model Loaded from {}".format(self.model_dir))
        else:
            print("No model to load")


def psnr(x, y):
    mse = np.mean(np.square(np.subtract(x, y)))
    if mse == 0:
        return 100
    else:
        return 10 * np.log10(1. / mse)


def load_matlab_data(train_dir, colour_format):
    if colour_format == 'ych':
        train_dir = train_dir + '/train_91_ychannels_matlab.h5'
    elif colour_format == 'ycbcr':
        train_dir = train_dir + '/train_91_ycbcrchannels_matlab.h5'
    elif colour_format == 'rgb':
        train_dir = train_dir + '/train_91_rgbchannels_matlab.h5'
    with h5py.File(train_dir, 'r') as f:
        x = np.array(f.get('data'))
        y = np.array(f.get('label'))
        return x, y


def load_octave_data(train_dir, colour_format):
    if colour_format == 'ych':
        train_dir = train_dir + '/train_91_ychannels_octave.h5'
    elif colour_format == 'ycbcr':
        train_dir = train_dir + '/train_91_ycbcrchannels_octave.h5'
    elif colour_format == 'rgb':
        train_dir = train_dir + '/train_91_rgbchannels_octave.h5'
    with h5py.File(train_dir, 'r') as f:
        x = np.array(f.get('data').get('value'))
        y = np.array(f.get('label').get('value'))
        return x, y


def imread(path):
    return Image.open(path)


def make_matlab_bc_data(train_dir, scale, colour_format):
    scale = scale
    path = train_dir
    if colour_format == 'ych' or 'ycbcr':
        if scale == 2:
            lr_us_path = path + '_2x_upscaled_mat_ycbcr/'
        elif scale == 3:
            lr_us_path = path + '_3x_upscaled_mat_ycbcr/'
        elif scale == 4:
            lr_us_path = path + '_4x_upscaled_mat_ycbcr/'
        else:
            print("Invalid value for scale")
    elif colour_format == 'rgb':
        if scale == 2:
            lr_us_path = path + '_2x_upscaled_mat_rgb/'
        elif scale == 3:
            lr_us_path = path + '_3x_upscaled_mat_rgb/'
        elif scale == 4:
            lr_us_path = path + '_4x_upscaled_mat_rgb/'
        else:
            print("Invalid value for scale")
    dir_list = os.listdir(lr_us_path)
    x = []
    y = []
    count = 0
    for file in dir_list:
        count += 1
        x_ = imread(os.path.join(lr_us_path, file))
        y_ = imread(os.path.join(lr_us_path, file))
        x_ = np.array(x_)
        y_ = np.array(y_)
        x.append(x_ / 255.)
        y.append(y_ / 255.)
    return x, y


def make_octave_bc_data(train_dir, scale, colour_format):
    scale = scale
    path = train_dir
    if colour_format == 'ych' or 'ycbcr':
        if scale == 2:
            lr_us_path = path + '_2x_upscaled_oct_ycbcr/'
        elif scale == 3:
            lr_us_path = path + '_3x_upscaled_oct_ycbcr/'
        elif scale == 4:
            lr_us_path = path + '_4x_upscaled_oct_ycbcr/'
        else:
            print("Invalid value for scale")
    elif colour_format == 'rgb':
        if scale == 2:
            lr_us_path = path + '_2x_upscaled_oct_rgb/'
        elif scale == 3:
            lr_us_path = path + '_3x_upscaled_oct_rgb/'
        elif scale == 4:
            lr_us_path = path + '_4x_upscaled_oct_rgb/'
        else:
            print("Invalid value for scale")
    dir_list = os.listdir(lr_us_path)
    x = []
    y = []
    count = 0
    for file in dir_list:
        count += 1
        x_ = imread(os.path.join(lr_us_path, file))
        y_ = imread(os.path.join(lr_us_path, file))
        x_ = np.array(x_)
        y_ = np.array(y_)
        x.append(x_ / 255.)
        y.append(y_ / 255.)
    return x, y


def make_matlab_gt_data(train_dir, colour_format):
    path = train_dir
    if colour_format == 'ych' or 'ycbcr':
        gt_path = path + '_gt_mat_ycbcr/'
    elif colour_format == 'rgb':
        gt_path = path + '_gt_mat_rgb/'
    dir_list = os.listdir(gt_path)
    y = []
    count = 0
    for file in dir_list:
        count += 1
        y_ = imread(os.path.join(gt_path, file))
        y_ = np.array(y_)
        y.append(y_ / 255.)
    return y


def make_octave_gt_data(train_dir, colour_format):
    path = train_dir
    if colour_format == 'ych' or 'ycbcr':
        gt_path = path + '_gt_oct_ycbcr/'
    elif colour_format == 'rgb':
        gt_path = path + '_gt_oct_rgb/'
    dir_list = os.listdir(gt_path)
    y = []
    count = 0
    for file in dir_list:
        count += 1
        y_ = imread(os.path.join(gt_path, file))
        y_ = np.array(y_)
        y.append(y_ / 255.)
    return y


def imsave_ycbcr(img, path, filename):
    img = img * 255.
    img = Image.fromarray(img.astype('uint8'), mode='YCbCr')
    img = img.convert('RGB')
    return img.save(os.path.join(path, filename))


def imsave_rgb(img, path, filename):
    img = img * 255.
    img = Image.fromarray(img.astype('uint8'), mode='RGB')
    return img.save(os.path.join(path, filename))


def save_res(path, gt, bc, srcnn, i, colour_format):
    if colour_format == 'ych' or 'ycbcr':
        imsave_ycbcr(gt, path, str(i) + '_gt.bmp')
        imsave_ycbcr(bc, path, str(i) + '_bc.bmp')
        imsave_ycbcr(srcnn, path, str(i) + '_srcnn.bmp')
    elif colour_format == 'rgb':
        imsave_rgb(gt, path, str(i) + '_gt.bmp')
        imsave_rgb(bc, path, str(i) + '_bc.bmp')
        imsave_rgb(srcnn, path, str(i) + '_srcnn.bmp')
    else:
        print("Improper value for colour_format")


def border_crop(img):
    padding = int((5 + 9 + 1 - 3) / 2)
    if img.ndim == 3:
        h, w, c = img.shape
        return img[padding:h - padding, padding:w - padding, :]
    else:
        h, w = img.shape
        return img[padding:h - padding, padding:w - padding]
