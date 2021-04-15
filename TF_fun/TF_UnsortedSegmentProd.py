import tensorflow as tf

a = tf.constant([[5, 2, 3, 4],
                 [5, 6, 7, 8],
                 [4, 3, 2, 1],
                 [1, 2, 3, 4]])
b = tf.constant([1, 0, 1, 1])
c = tf.raw_ops.UnsortedSegmentProd(data=a, segment_ids=b, num_segments=4)


"""
输入两个tensor，分组计算tensor_a乘积
num_segments >= 2
b列向量维度与a_tensor的维度相等
"""

if __name__ == '__main__':
    with tf.Session() as sess:
        print(sess.run(c))

print('over')