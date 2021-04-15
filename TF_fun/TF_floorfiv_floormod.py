import tensorflow as tf

a=tf.constant([7,6,6,8])
b=tf.constant([2,5,3,4])
c=tf.floordiv(a,b)
d=tf.floormod(a,b)
with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d))