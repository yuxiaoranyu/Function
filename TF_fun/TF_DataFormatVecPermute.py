import tensorflow as tf

x_tensor=tf.constant([2,3,4,5])
DF=tf.raw_ops.DataFormatVecPermute(x=x_tensor)
with tf.Session() as sess:
    print(sess.run(DF))