import tensorflow as tf

'''
矩阵对角线
拉伸矩阵对角线
'''
a=tf.constant([[1,3,4,6],
               [2,5,8,8],
               [4,6,9,7]])

b=tf.raw_ops.MatrixDiagPartV2(input=a,k=(0,3),padding_value=0)
c=tf.constant([[1,2,3,4],
               [5,6,7,8]])
d=tf.raw_ops.MatrixDiagV2(diagonal=c,k=(0,1),num_cols=8,num_rows=4,padding_value=0)
m=tf.matrix_diag(diagonal=c)
with tf.Session() as sess:
    print(sess.run(b))
    print(sess.run(d))
    # print(sess.run(m))