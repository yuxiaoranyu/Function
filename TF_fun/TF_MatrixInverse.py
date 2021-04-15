import tensorflow as tf

'''
求逆矩阵
'''

a=tf.constant([[3,2],
               [4,4]],dtype=tf.float32)

b=tf.raw_ops.MatrixInverse(input=a)

c=tf.matmul(a,b)
with tf.Session() as sess:
    print(sess.run(b))
    print(sess.run(c))
