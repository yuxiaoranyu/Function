import tensorflow as tf

a_indices = tf.constant([[2, 1],
                         [2, 2]], dtype=tf.int64)
a_values = tf.constant([4, 8])
a_shape = tf.constant([3, 3], dtype=tf.int64)
b_indices = tf.constant([[1, 3],
                         [1, 2]], dtype=tf.int64)
b_values = tf.constant([3, 5])
b_shape = tf.constant([3, 3], dtype=tf.int64)
# thresh=
# tf.sparse_add
sum_indices, sum_values, sum_shape = tf.raw_ops.SparseAdd(a_indices=a_indices, a_values=a_values, a_shape=a_shape,
                                                          b_indices=b_indices, b_values=b_values, b_shape=b_shape,
                                                          thresh=0.21)
backprop_val_grad = tf.constant([8,6,4,7])
sum_in=tf.constant([[2,2],
                    [2,1],
                    [1,2],
                    [1,3]],dtype=tf.int64)
a_val_grad, b_val_grad = tf.raw_ops.SparseAddGrad(backprop_val_grad=backprop_val_grad, a_indices=a_indices,
                                                  b_indices=b_indices, sum_indices=sum_in)

with tf.Session() as sess:
    print(sess.run(sum_indices))
    print(sess.run(sum_values))
    print(sess.run(sum_shape))
    print(sess.run(a_val_grad))
    print(sess.run(b_val_grad))
