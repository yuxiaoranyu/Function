import tensorflow as tf

a=tf.constant([1,2,3])
#泊松分布
b=tf.raw_ops.RandomPoissonV2(shape=a,rate=0.1)

with tf.Session() as sess:
    print(sess.run(b))
