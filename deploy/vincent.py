import tensorflow as tf
import tensorflow.contrib.layers as ly
import numpy as np
import time


class vincent:
    def __init__(self, r=0., lr=1e-4, vgg19_npy_path=None):
        self.decoder_data_dict = np.load('decoder.npy', encoding='latin1').item()
        print("npy file loaded")
        self.r = r
        self.lr = lr

    def build(self, content, style):
        start_time = time.time()
        print("build model started")
        content = content / 255.0
        style = style / 255.0
        self.feature_con = self.extractor(content, name='extractor_con')
        self.feature_sty = self.extractor(style, name='extractor_sty')

        self.mean = tf.reduce_mean(self.feature_sty, axis=[0, 1, 2])
        self.std = tf.reduce_mean(tf.abs(self.feature_sty - self.mean),
                                  axis=[0, 1, 2])
        self.mean, self.var = tf.nn.moments(self.feature_sty, [0, 1, 2])
        self.std = tf.sqrt(1e-16 + self.var)
        '''
        self.mean, self.var = tf.nn.moments(self.feature_sty, [0, 1, 2])
        print 'tmp', self.ten_sh(tmp)
        self.std = tf.sqrt(self.var+1e-16)
        self.mean = tf.squeeze(self.mean)
        self.std = tf.squeeze(self.std)
        '''
        print 'mean', self.ten_sh(self.mean)
        # Adaptive Instance Normalization
        self.AdaIn = tf.nn.batch_normalization(x=self.feature_con,
                                               mean=self.mean,
                                               variance=self.std,
                                               offset=self.mean,
                                               scale=self.std,
                                               variance_epsilon=1e-5
                                               )
        '''
        self.tconv4_1 = self.tconv_ly(self.AdaIn, 512, 256, 'tc4_1')
        self.up3 = self.up_sampling(self.tconv4_1, 'up_3')
        self.tconv3_4 = self.tconv_ly(self.up3, 256, 256, 'tc3_4')
        self.tconv3_3 = self.tconv_ly(self.tconv3_4, 256, 256, 'tc3_3')
        self.tconv3_2 = self.tconv_ly(self.tconv3_3, 256, 256, 'tc3_2')
        self.tconv3_1 = self.tconv_ly(self.tconv3_2, 256, 128, 'tc3_1')
        self.up2 = self.up_sampling(self.tconv3_1, 'up_2')
        self.tconv2_2 = self.tconv_ly(self.up2, 128, 128, 'tc2_2')
        self.tconv2_1 = self.tconv_ly(self.tconv2_2, 128, 64, 'tc2_1')
        self.up1 = self.up_sampling(self.tconv2_1, 'up_1')
        self.tconv1_2 = self.tconv_ly(self.up1, 64, 64, 'tc1_2')
        self.tconv1_1 = self.tconv_ly(self.tconv1_2, 64, 3, 'tc1_1')
        '''
        self.tconv1_1 = self.decoder(self.AdaIn, 'decoder')

        self.output = self.tconv1_1*255.0
        
        # loss
        '''
        self.feature_2 = self.extractor(self.tconv1_1, name='extractor_2')
        self.content_loss = self.MSE(self.feature_2['conv4_1'], self.AdaIn)
        style_loss_list = []
        for fea_1, fea_2 in zip(self.feature_1.values(),
                                self.feature_2.values()):
            mean_1, var_1 = tf.nn.moments(tf.stack(tf.unpack(fea_1)[1:]),
                                          [1, 2])
            std_1 = tf.sqrt(var_1)
            print 'mean_1', self.ten_sh(mean_1)
            mean_2, var_2 = tf.nn.moments(fea_2, [1, 2])
            std_2 = tf.sqrt(var_2)
            print 'mean_2', self.ten_sh(mean_2)
            style_loss_list.append(self.MSE(mean_1, mean_2))
            style_loss_list.append(self.MSE(std_1, std_2))

        print(("build model finished: %ds" % (time.time() - start_time)))

        self.style_loss = tf.add_n(style_loss_list)
        self.loss = self.content_loss + tf.multiply(self.r, self.style_loss)
        #optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        print(("build model finished: %ds" % (time.time() - start_time)))
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        #optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr)
        #self.opt = optimizer.minimize(self.loss)
        #self.debug_opt = optimizer.minimize(self.debug_loss)
        '''
        print(("build model finished: %ds" % (time.time() - start_time)))

    def extractor(self, image, name):
        with tf.variable_scope(name):
            conv0_1 = self.tconv_layer(image, "conv0_1")
            conv1_1 = self.tconv_layer(conv0_1, "conv1_1")
            conv1_2 = self.tconv_layer(conv1_1, "conv1_2")
            up_1 = self.max_pool(conv1_2, 'up_1')

            conv2_1 = self.tconv_layer(up_1, "conv2_1")
            conv2_2 = self.tconv_layer(conv2_1, "conv2_2")
            up_2 = self.max_pool(conv2_2, 'up_2')

            conv3_1 = self.tconv_layer(up_2, "conv3_1")
            conv3_2 = self.tconv_layer(conv3_1, "conv3_2")
            conv3_3 = self.tconv_layer(conv3_2, "conv3_3")
            conv3_4 = self.tconv_layer(conv3_3, "conv3_4")

            up_3 = self.max_pool(conv3_4, 'up_3')
            conv4_1 = self.tconv_layer(up_3, "conv3_5")

            return conv4_1


    def decoder(self, fea, name):
        with tf.variable_scope(name):
            tconv1_1 = self.tconv_layer(fea, "tconv1_1")
            up_1 = self.up_sampling(tconv1_1, 'up_1')
            tconv2_1 = self.tconv_layer(up_1, "tconv2_1")
            tconv2_2 = self.tconv_layer(tconv2_1, "tconv2_2")
            tconv2_3 = self.tconv_layer(tconv2_2, "tconv2_3")
            tconv2_4 = self.tconv_layer(tconv2_3, "tconv2_4")
            up_2 = self.up_sampling(tconv2_4, 'up_2')
            tconv3_1 = self.tconv_layer(up_2, "tconv3_1")
            tconv3_2 = self.tconv_layer(tconv3_1, "tconv3_2")
            up_3 = self.up_sampling(tconv3_2, 'up_3')
            tconv4_1 = self.tconv_layer(up_3, "tconv4_1")
            tconv4_2 = self.tconv_layer(tconv4_1, "tconv4_2")
            return tconv4_2

    def MSE(self, target, pred):
        return tf.reduce_mean(tf.abs(tf.subtract(target, pred)))

    def up_sampling(self, bottom, name=None):
        size = self.ten_sh(bottom)[1:3]
        return tf.image.resize_nearest_neighbor(bottom, np.multiply(size, 2))

    def avg_pool(self, bottom, name=None):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name=None):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME', name=name)

    def tconv_ly(self, bottom, in_channels, out_channels, name=None):
        with tf.variable_scope(name):
            shape = self.ten_sh(bottom)
            shape[-1] = out_channels
            bottom = tf.pad(bottom,
                            [[0, 0], [2, 2], [2, 2], [0, 0]], "REFLECT")
            init = ly.xavier_initializer_conv2d()
            output = ly.convolution2d_transpose(inputs=bottom,
                                                num_outputs=out_channels,
                                                kernel_size=[3, 3],
                                                stride=[1, 1],
                                                padding='VALID',
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=init,
                                                biases_initializer=init,
                                                trainable=True)
            output = tf.slice(output, begin=[0, 2, 2, 0], size=shape)
            #crop_fn = tf.image.resize_image_with_crop_or_pad
            #crop_lam = lambda img: crop_fn(img, target_height=shape[1],
                                           #target_width=shape[2])
            #output = tf.map_fn(crop_lam, output, parallel_iterations=shape[0])
            #output = crop_fn(output, target_height=shape[1], target_width=shape[2])
            print self.ten_sh(output)

            return output

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def tconv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            filt = tf.transpose(filt, perm=[2, 3, 1, 0])
            if name != 'conv0_1':
                bottom = tf.pad(bottom,
                                [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            if name != 'conv0_1':
                bias = tf.nn.relu(bias)
            return bias

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
        try:
            return tf.constant(self.data_dict[name]['weights'], name="filter")
        except:
            return tf.constant(np.float32(self.decoder_data_dict[name]['weights']), name="filter")

    def get_bias(self, name):
        try:
            return tf.constant(self.data_dict[name]['biases'], name="biases")
        except:
            return tf.constant(np.float32(self.decoder_data_dict[name]['biases']), name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

    def ten_sh(self, tensor):
        return tensor.get_shape().as_list()
