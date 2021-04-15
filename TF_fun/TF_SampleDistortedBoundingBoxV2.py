import tensorflow as tf
from matplotlib import pyplot as plt

img_path=r'../jiejie.jpg'

img_raw=tf.gfile.FastGFile(img_path,'rb').read()
# print(img_raw)
img=tf.image.decode_jpeg(img_raw)

a=tf.constant([[[3,3,3],
                [1,2,3]],
               [[1,2,3],
                [1,2,3]],
               [[1, 2, 3],
                [1, 2, 3]]
               ])
b_box=tf.constant([[[1,2,3]],
                   [[2,3,4]]],dtype=tf.float32)
m_o_c=tf.constant([[1,2,3,4]],dtype=tf.float32)


with tf.Session() as sess:
    img = img.eval()
    img = tf.cast(img, tf.float32)
    # print(img.dtype, img.shape)

    begin, size, bbox_for_draw = tf.raw_ops.SampleDistortedBoundingBoxV2(image_size=[16,16], bounding_boxes=0.5,
                                                                         min_object_covered=0.1)
    print(sess.run(begin))
    print(sess.run(bbox_for_draw))
    print(sess.run(size))