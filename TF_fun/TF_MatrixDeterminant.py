import tensorflow as tf

#计算行列式
a=tf.constant([[1,4],
               [4,4]],dtype=tf.float32)
b=tf.raw_ops.MatrixDeterminant(input=a)

with tf.Session() as sess:
    print(sess.run(b))