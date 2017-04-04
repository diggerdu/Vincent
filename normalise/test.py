from vincent import vincent
import tensorflow as tf
import numpy as np
np.set_printoptions(precision=6, suppress=True)
import PIL
from PIL import Image as Im

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train_iters = 1000000
eval_step = 100


content = np.uint8(np.array(Im.open('content.jpg').resize((512, 512),
                   PIL.Image.ANTIALIAS)))
style = np.uint8(np.array(Im.open('style.jpg').crop((0, 0, 512, 512))))
V = vincent(vgg19_npy_path='vgg19_normal.npy')
rgb = tf.placeholder(tf.float32, shape=[2, 512, 512, 3])
V.build(rgb)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 1
f_d = {rgb: np.asarray([content, style])}

output = sess.run(V.output, feed_dict=f_d)
fea = sess.run(V.feature_sty, feed_dict=f_d)
print 'feature_shape', fea.shape
np.save('debug.npy', fea)

std = sess.run(V.std, feed_dict=f_d)
print std.shape
print 'mean', std

output = np.squeeze(output)
print np.mean(output)
Im.fromarray(np.uint8(output)).save('compose.png')
