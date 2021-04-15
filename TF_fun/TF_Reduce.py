import tensorflow as tf

a=tf.constant([[[[1,2,9,4],
                [3,5,6,8]]]])
print(a.shape)
b=tf.reduce_max(a,axis=-1)
c=tf.reduce_min(a,axis=0)
d=tf.reduce_mean(a,axis=1)
prod=tf.reduce_prod(a,axis=3)
Sum=tf.reduce_sum(a,axis=2)

with tf.Session() as sess:
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(d))
    print(sess.run(prod))
    print(sess.run(Sum))