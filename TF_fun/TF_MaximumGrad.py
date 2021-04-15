import tensorflow as tf

a=tf.constant([4,2,4,8])
b=tf.constant([2,5,8,9])
c=tf.maximum(a,b)
d=tf.minimum(a,b)
with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d))