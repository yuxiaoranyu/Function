import tensorflow as tf



x=tf.constant([0.5],dtype=tf.float32)
a=tf.constant([2],dtype=tf.float32)
b=tf.constant([2],dtype=tf.float32)
#x=[0,1]
Betainc=tf.raw_ops.Betainc(a=2.0,b=2.0,x=0.5)
with tf.Session() as sess:
    print(sess.run(Betainc))

