import tensorflow as tf
import numpy as np
a=np.array([[1,6,3],
               [5,4,9],
               [1,2,3],
               [7,5,8]],dtype=np.int64)
input_values=np.array([5,6,3,4],dtype=np.int64)
input_shape=np.array([3,9,12],dtype=np.int64)
reduction_axes=np.array([0],dtype=np.int64)
print(a.shape)
b=tf.raw_ops.SparseReduceSum(input_indices=a, input_values=input_values, input_shape=input_shape, reduction_axes=reduction_axes)
with tf.Session() as sess:
    print(sess.run(b))

'''
input_indices 位于稀疏矩阵中的位置
input_values  稀疏矩阵中的值
input_shape 稀疏矩阵的形状，用0填充
reduction_axes 输出稀疏矩阵的维度
'''