import tensorflow as tf
from matplotlib import pyplot as plt

img_path = r'../jiejie.jpg'

img_raw = tf.gfile.FastGFile(img_path, 'rb').read()
# print(img_raw)
img = tf.image.decode_jpeg(img_raw)

with tf.Session() as sess:
    img = img.eval()
    img = tf.cast(img, tf.float32)
    print(img.dtype, img.shape)
    img=tf.reshape(img,(1,2434,3583,3))

    size=tf.constant([1600,1600])
    scale=tf.constant([800,800],dtype=tf.float32)
    translation=tf.constant([400,600],dtype=tf.float32)
    SAT=tf.raw_ops.ScaleAndTranslate(images=img, size=size, scale=scale, translation=translation)
    tf.raw_ops.ScaleAndTranslateGrad
    SAT=tf.reshape(SAT,(1600,1600,3))
    SAT = tf.cast(SAT, tf.uint8)
    img_ = SAT.eval()
    print(img_.shape)
    # plt.figure(1)
    plt.imshow(img_)
    plt.show()