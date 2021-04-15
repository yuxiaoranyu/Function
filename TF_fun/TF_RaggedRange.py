import tensorflow as tf


starts=tf.constant([1,5,8])
limits=tf.constant([5,10,14])
rt_nested_splits,rt_dense_values=tf.raw_ops.RaggedRange(starts=starts, limits=limits, deltas=2,)
with tf.Session() as sess:
    print(sess.run(rt_nested_splits))
    print(sess.run(rt_dense_values))
