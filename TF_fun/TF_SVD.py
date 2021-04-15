import tensorflow as tf

a=tf.constant([[2,3],
               [4,5]],dtype=tf.float32)

s,v,d=tf.raw_ops.Svd(input=a)
with tf.Session() as sess:
    print(sess.run(s))
    print(sess.run(v))
    print(sess.run(d))
