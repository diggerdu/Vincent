

from PIL import Image as Im
import numpy as np
import tensorflow as tf
from vincent import vincent

V = vincent()
content_holder = tf.placeholder(tf.float32, shape=[8, 512, 512, 3])
style_holder = tf.placeholder(tf.float32, shape=[8, 512, 512, 3])
V.build(content_holder, style_holder)

