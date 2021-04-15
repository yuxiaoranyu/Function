import tensorflow as tf
import numpy as np

shape=tf.constant([1,3,3])
ParameterizedTruncatedNormal=tf.raw_ops.ParameterizedTruncatedNormal(shape=shape, means=1.0, stdevs=1.0, minvals=2.0, maxvals=6.0)
with tf.Session() as sess:
    x = sess.run(ParameterizedTruncatedNormal)
    print(x)
    a=np.mean(x)
    print(a)