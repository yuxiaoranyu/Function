import tensorflow as tf

a=tf.constant([[1,2,8],
               [3,4,6],
               [5,6,7]])

b=tf.raw_ops.RandomShuffle(value=a,seed=1,seed2=2)

with tf.Session() as sess:
    print(sess.run(b))