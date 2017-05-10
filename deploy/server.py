import os
import io
from flask import Flask, request, send_file


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

app = Flask(__name__)


@app.route('/', methods=['post'])
def transfer():
    style_img = request.files['style']
    content_img = request.files['content']
    style = np.uint8(np.array(Im.open(style_img)))[..., 0:3]
    content = np.uint8(np.array(Im.open(content_img)))[..., 0:3]
    f_d = {content_holder: np.asarray([content]),
           style_holder: np.asarray([style])}
    output = np.squeeze(sess.run(V.output, feed_dict=f_d))
    Im.fromarray(np.uint8(output)).save("/tmp/tmp.png")
    print 'OK'
    return send_file('/tmp/tmp.png')

if __name__ == '__main__':
    app.run(port=42513)
