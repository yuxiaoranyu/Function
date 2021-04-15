import tensorflow as tf


input_=tf.constant( [[-5, 100,62],
                     [150, 10,45],
                     [5, 100,250]])
boundaries=[0,50,100,200]
Bu=tf.raw_ops.Bucketize(input=input_,boundaries=boundaries)
with tf.Session() as sess:
    print(sess.run(Bu))