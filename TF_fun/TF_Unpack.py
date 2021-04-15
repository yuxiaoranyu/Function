import tensorflow as tf

a=tf.constant([[1,2,3,4],
               [4,5,6,5],
               [4,6,9,8]])
b=tf.raw_ops.Unpack(value=a,num=4,axis=1)
with tf.Session() as sess:
    print(sess.run(b))
