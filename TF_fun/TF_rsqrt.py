import tensorflow as tf

a=tf.constant([4],dtype=tf.float16)

b=tf.rsqrt(a)

with tf.Session() as sess:
    print(sess.run(b))
