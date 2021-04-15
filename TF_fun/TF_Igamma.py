import tensorflow as tf


a=tf.constant([2],dtype=tf.float32)
x=tf.constant([2],dtype=tf.float32)
igamma=tf.raw_ops.Igamma(a=a,x=x)
igammac=tf.raw_ops.Igammac(a=a,x=x)
igammagrad=tf.raw_ops.IgammaGradA(a=a,x=x)
with tf.Session() as sess:
    print(sess.run(igamma))
    print(sess.run(igammac))
    print(sess.run(igammagrad))