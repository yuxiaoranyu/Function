import tensorflow as tf

a=tf.constant([1,2,3,4])

b=tf.raw_ops.RandomUniformInt(shape=a,minval=1,maxval=3)
with tf.Session() as sess:
    print(sess.run(b))