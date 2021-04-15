import tensorflow as tf
import numpy as np

# np.bincount
# tf.bincount
arr=tf.constant([[1,2,9,9],
                 [4,1,5,9]],dtype=tf.int32)

size=tf.constant([5])
weights=tf.constant([7,2,2,9],dtype=tf.int32)
b=tf.raw_ops.Bincount(arr=arr,size=8,weights=[])
with tf.Session() as sess:
    print(sess.run(b))