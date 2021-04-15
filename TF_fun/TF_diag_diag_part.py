import tensorflow as tf

a=tf.constant([6,2,3])

a_=tf.constant([[1,2,3],
                [4,5,6],
                [7,8,9]])
b=tf.diag(a)
c=tf.diag_part(a_)
with tf.Session() as sess:
    print(sess.run(b))
    print(sess.run(c))

