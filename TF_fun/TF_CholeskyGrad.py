import tensorflow as tf

tensor=tf.constant([[3,8,6],
                    [1,2,9],
                    [8,6,3]],dtype=tf.float32)
l=tf.constant([[8,2,4],
               [4,8,2],
               [9,7,2]],dtype=tf.float32)
Cholesky=tf.raw_ops.CholeskyGrad(l=l,grad=tensor)
with tf.Session() as sess:
    print(sess.run(Cholesky))