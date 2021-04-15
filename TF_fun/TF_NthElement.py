import tensorflow as tf

input=tf.constant([[2,3,4,9],
                   [5,8,9,6],
                   [4,2,12,6]])

Nth=tf.raw_ops.NthElement(input=input, n=3,reverse=True)
with tf.Session() as sess:
    print(sess.run(Nth))