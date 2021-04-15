import tensorflow as tf

a=tf.constant([1],dtype=tf.float32)
sh=tf.constant([2])
b=tf.raw_ops.RandomGamma(shape=sh,alpha=a)
c=tf.raw_ops.RandomGammaGrad(sample=2,alpha=a)

with tf.Session() as sess:
    print(sess.run(b))
    print(sess.run(c))
