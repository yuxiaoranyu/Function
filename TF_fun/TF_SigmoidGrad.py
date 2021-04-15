import tensorflow as tf

a=tf.constant([1,2,3,4],dtype=tf.float16)

b=tf.raw_ops.SigmoidGrad(y=a,dy=1)
with tf.Session() as sess:
    print(sess.run(b))


