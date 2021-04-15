import tensorflow as tf

input_tensor=tf.constant([[[1,1,1,1],
                           [1,1,1,1],
                           [1,1,1,1]],
                          [[1,2,3,1],
                           [2,3,4,1],
                           [6,8,1,1]],
                          [[3,4,5,1],
                           [9,9,7,1],
                           [4,5,6,1]]])
diagonal=tf.constant([[2,3,6],
                      [4,5,9],
                      [2,3,4]])
BandPart=tf.raw_ops.MatrixBandPart(input=input_tensor, num_lower=-1, num_upper=2)
MatrixSetDiagV2=tf.raw_ops.MatrixSetDiagV2(input=input_tensor,diagonal=diagonal,k=1)
# a=tf.matrix_set_diag(diagonal=diagonal)

with tf.Session() as sess:
    # print(sess.run(BandPart))
    print(sess.run(MatrixSetDiagV2))
    # print(sess.run(a))