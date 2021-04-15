import tensorflow as tf

a=tf.constant([[[[1],
               [2]]]])
print(a.shape)
b=tf.raw_ops.Squeeze(input=a,axis=[3])
print(b.shape)
with tf.Session() as sess:
    print(sess.run(b))