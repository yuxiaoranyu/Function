import tensorflow as tf
import numpy as np

a_=np.inf
print(a_)
a=tf.constant([2,np.inf,9,np.nan],dtype=tf.float16)
b=tf.is_inf(a)
c=tf.is_nan(a)

with tf.Session() as sess:
    print(sess.run(b))
    print(sess.run(c))
