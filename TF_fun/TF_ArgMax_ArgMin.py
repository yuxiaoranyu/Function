import tensorflow as tf

a = tf.constant([[1, 3, 5, 9],
                 [2, 8, 10, 6],
                 [7, 9, 5, 4]])
b = tf.argmax(a,axis=1)
c = tf.argmin(a,axis=0)
with tf.Session() as sess:
    print(sess.run(b))
    print(sess.run(c))
