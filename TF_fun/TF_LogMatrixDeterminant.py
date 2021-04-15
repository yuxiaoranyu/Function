import tensorflow as tf

a=tf.constant([[1,2],
               [3,4]],dtype=tf.float32)

d=tf.constant([[1,2],
               [3,4]],dtype=tf.float32)
m=tf.matmul(a,d)
# print(m)
b,c=tf.raw_ops.LogMatrixDeterminant(input=a)
with tf.Session() as sess:
    print(sess.run(b))#符号位
    print(sess.run(c))#行列式的绝对值的对数
    print(sess.run(m))






