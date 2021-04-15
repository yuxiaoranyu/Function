import tensorflow as tf

x = tf.constant([-4, -2, 2, 3, 8, 6, 7])
y = tf.constant([2, -2, 0, 4, 9, 2, -2])
RightShift = tf.raw_ops.RightShift(x=x, y=y)
with tf.Session() as sess:
    print(sess.run(RightShift))
