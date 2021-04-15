import tensorflow as tf


logits=tf.constant([[3,6,4],
                    [4,10,9]],dtype=tf.float32)

Multinomial=tf.raw_ops.Multinomial(logits=logits, num_samples=4)
with tf.Session() as sess:
    print(sess.run(Multinomial))