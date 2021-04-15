import tensorflow as tf

a=tf.constant([[2,2,5],
               [2,4,8],
               [7,9,4]],dtype=tf.float32)
rhs_=tf.constant([[3],
                  [1],
                  [3]],dtype=tf.float32)
b=tf.raw_ops.MatrixTriangularSolve(matrix=a,rhs=rhs_,lower=True)

with tf.Session() as sess:
    print(sess.run(b))