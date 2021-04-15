import tensorflow as tf

a=tf.constant([[5,6,3,6],
               [7,8,6,4]])
b=tf.constant([[2,8,3,5],
               [5,9,6,8]])
c=tf.greater(a,b)
d=tf.greater_equal(a,b)
with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d))