import tensorflow as tf


matrix=tf.constant([[2,4,8],
                    [4,2,6],
                    [6,5,6]],dtype=tf.float32)
rhs=tf.constant([[4],
                 [2],
                 [3]],dtype=tf.float32)

input=tf.constant([[1.0,4.6],
                   [9.0,16.0]],dtype=tf.complex64)

MatrixSolveLs=tf.raw_ops.MatrixSolveLs(matrix=matrix, rhs=rhs, l2_regularizer=1.0)
MatrixSquareRoot=tf.raw_ops.MatrixSquareRoot(input=input)
with tf.Session() as sess:
    print(sess.run(MatrixSolveLs))
    print(sess.run(MatrixSquareRoot))