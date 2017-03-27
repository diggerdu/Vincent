import os
import tensorflow as tf
import tensorflow.contrib.layers as ly
import numpy as np
import time
import inspect

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is None:
            path = inspect.getfile(Vgg19)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg19.npy")
            vgg19_npy_path = path
            print(vgg19_npy_path)

        self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        print("npy file loaded")
        self.init = ly.xavier_initializer_conv2d()

    def build(self, rgb):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.feature_con, self.feature_sty = tf.unpack(self.conv4_1, axis=0)
        self.mean, self.var = tf.nn.moments(self.feature_sty, [0, 1, 2])
        # Adaptive Instance Normalization
        self.AdaIn = tf.nn.batch_normalization(x=self.feature_con,
                                               mean=self.mean,
                                               variance=self.var,
                                               offset=self.mean,
                                               scale=self.var,
                                               variance_epsilon=1e-6
                                               )
        # gc
        self.data_dict = None

        self.tconv4_1 = self.conv_tran_layer(self.AdaIn, 512, 256)
        self.up3 = self.up_sampling(self.tconv4_1)
        self.tconv3_4 = self.conv_tran_layer(self.up3, 256, 256)
        self.tconv3_3 = self.conv_tran_layer(self.tconv3_4, 256, 256)
        self.tconv3_2 = self.conv_tran_layer(self.tconv3_3, 256, 256)
        self.tconv3_1 = self.conv_tran_layer(self.tconv3_2, 256, 128)
        self.up2 = self.up_sampling(self.tconv3_1)
        self.tconv2_2 = self.conv_tran_layer(self.up2, 128, 128)
        self.tconv2_1 = self.conv_tran_layer(self.tconv2_2, 128, 64)
        self.up1 = self.up_sampling(self.tconv2_1)
        self.tconv2_2 = self.conv_tran_layer(self.tconv2_2, 64, 64)
        self.tconv2_1 = self.conv_tran_layer(self.tconv2_2, 64, 3)

        print(("build model finished: %ds" % (time.time() - start_time)))
    def up_sampling(self, bottom, name=None):
        size = self.ten_sh(bottom)
        return tf.image.resize_nearest_neighbor(bottom, size*2)
    def avg_pool(self, bottom, name=None):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name=None):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_tran_layer(self, bottom, in_channels, out_channels, name=None):
        with tf.variable_scope(name):
            init = ly.xavier_initializer_conv2d()
            output = ly.convolution2d_transpose(inputs=bottom,
                                                num_outputs=out_channels,
                                                kernel_size=[3, 3],
                                                stride=[1, 1],
                                                padding='SAME',
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=init,
                                                bias_initializer=init,
                                                trainable=True
                                                )

            return output

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

    def ten_sh(self, tensor):
        return tensor.get_shape().as_list()
