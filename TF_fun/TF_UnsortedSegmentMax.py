import tensorflow as tf

a = tf.constant([[1, 2, 3, 6],
                 [2, 3, 4, 3],
                 [9, 8, 7, 5],
                 [7, 5, 9, 10]])
b = tf.constant([0, 1, 1, 1])
c = tf.raw_ops.UnsortedSegmentMax(data=a, segment_ids=b, num_segments=3)
d = tf.raw_ops.UnsortedSegmentMin(data=a, segment_ids=b, num_segments=3)
prod = tf.raw_ops.UnsortedSegmentProd(data=a, segment_ids=b, num_segments=4)
un_sum=tf.raw_ops.UnsortedSegmentSum(data=a,segment_ids=b,num_segments=3)

with tf.Session() as sess:
    print(sess.run(c))
    print(sess.run(d))
    print(sess.run(prod))
    print(sess.run(un_sum))
