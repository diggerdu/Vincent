from PIL import Image as Im
import numpy as np
import tensorflow as tf
from vincent import vincent

V = vincent()
content_holder = tf.placeholder(tf.float32, shape=[1, 512, 512, 3])
style_holder = tf.placeholder(tf.float32, shape=[1, 512, 512, 3])
V.build(content_holder, style_holder)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


def transfer(content_img, style_img):
    style = np.uint8(np.array(Im.open(style_img)))[..., 0:3][:,:,::-1]
    print style.shape
    content = np.uint8(np.array(Im.open(content_img)))[..., 0:3][:,:,::-1]
    print content.shape
    f_d = {content_holder: np.asarray([content]),
           style_holder: np.asarray([style])}
    output = np.squeeze(sess.run(V.output, feed_dict=f_d))[:,:,::-1]
    Im.fromarray(np.uint8(output)).save("output.png")
    Im.fromarray(np.uint8(output[:,:,::-1])).save("output_re.png")

if __name__ == '__main__':
    transfer('content.jpg', 'style.jpg')
