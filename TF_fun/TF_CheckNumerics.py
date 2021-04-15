import tensorflow as tf


tensor=tf.constant([[1.8,3.2,5.3,9.6],
                    [7.7,8.6,3.5,4.3]],dtype=tf.float32)
message=''
CN=tf.raw_ops.CheckNumerics(tensor=tensor,message=message)
with tf.Session() as sess:
    print(sess.run(CN))