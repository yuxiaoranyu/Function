import tensorflow as tf


a=tf.constant([[2,2,3],
               [5,2,5],
               [5,2,2]],dtype=tf.float32)
b,c=tf.raw_ops.Lu(input=a)

with tf.Session() as sess:
    print(sess.run(b))
    print(sess.run(c))#行索引
'''
input=LU
 | 5.   2.   5. |                |1,   0, 0|    |5, 2,   5|  |1|     |5,2,5| 
 | 0.4  1.2  1. |   ===>LU===>   |0.4, 1, 0|    |0, 1.2, 1|  |0|  =  |2,2,3|
 | 1.   0.  -3. |                |1,   0, 1|    |0, 0,  -3|  |2|     |5,2,2| 
[1 0 2]原矩阵的行索引
'''