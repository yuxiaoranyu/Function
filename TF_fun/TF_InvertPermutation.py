import tensorflow as tf


x=tf.constant([1,2,0,3,4])
invert=tf.raw_ops.InvertPermutation(x=x)
with tf.Session() as sess:
    print(sess.run(invert))

x = [3,4,0,2,1]
for i in range(len(x)):
    print("y[",x[i],"]",i)