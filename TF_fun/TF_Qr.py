import tensorflow as tf

a=tf.constant([[1,2,3],
               [3,2,4],
               [1,3,4]],dtype=tf.float32)
b,c=tf.raw_ops.Qr(input=a)
d=tf.matmul(b,c)
with tf.Session() as sess:
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(d))
