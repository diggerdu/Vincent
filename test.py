from vincent import vincent
import tensorflow as tf


V = vincent(vgg19_npy_path='vgg19.npy')
rgb = tf.placeholder(tf.float32, shape=[2, 224, 224, 3])
V.build(rgb)
