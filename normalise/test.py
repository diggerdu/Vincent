from vincent import vincent
import tensorflow as tf
import numpy as np
import PIL
from PIL import Image as Im

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train_iters = 1000000
eval_step = 100


content = np.uint8(np.array(Im.open('content.jpg').resize((224, 224),
                   PIL.Image.ANTIALIAS)))
style = np.uint8(np.array(Im.open('style.jpg').resize((224, 224),
                 PIL.Image.ANTIALIAS)))
V = vincent(vgg19_npy_path='vgg19_normal.npy')
rgb = tf.placeholder(tf.float32, shape=[2, 224, 224, 3])
V.build(rgb)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
step = 1
f_d = {rgb: np.asarray([content, style])}

while step < train_iters:
    ou, c_loss, s_loss, _  = sess.run([V.output, V.content_loss, V.style_loss, V.opt],
                             feed_dict=f_d)
    debug_rgb = sess.run(V.debug_rgb, feed_dict=f_d)
    print np.max(debug_rgb[1:,...])
    print 'ou', np.mean(ou)
    print ('at epoch {0}, content loss is {1}, style loss is {2}'.format(step, c_loss, s_loss))
#    debug = sess.run(V.feature_con, feed_dict=f_d)
#    print np.max(debug)
    if step % eval_step == 0:
        output = np.squeeze(sess.run(V.output, feed_dict=f_d))
        print np.max(output)
        print output[...,0]
        Im.fromarray(np.uint8(output)).save('output/{0}.png'.format(step))
    step += 1
