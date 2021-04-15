import tensorflow as tf

codebook_size = 50
codebook_bits = codebook_size * 32

input_tensor=tf.constant([2,4,6,8,10,12,14,64,2,4,6,8,10,12,14,64])
threod=tf.constant([4])
Com=tf.raw_ops.CompareAndBitpack(input=input_tensor,threshold=14)
x_tensor=tf.constant([1,-2],dtype=tf.complex64)
CAbs=tf.raw_ops.ComplexAbs(x=x_tensor)
with tf.Session() as sess:
    print(sess.run(Com))
    print(sess.run(CAbs))
