import tensorflow as tf

a=tf.constant([2,2,3,4],dtype=tf.int32)

b=tf.raw_ops.RandomStandardNormal(shape=a,dtype=tf.float32)

with tf.Session() as sess:
    print(sess.run(b))
