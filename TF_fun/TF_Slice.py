import tensorflow as tf

a=tf.constant([[1,2,3,4],
               [4,5,6,7],
               [3,4,5,6]])

b=tf.raw_ops.Slice(input=a,begin=[0,1],size=[3,3])
with tf.Session() as sess:
    print(sess.run(b))