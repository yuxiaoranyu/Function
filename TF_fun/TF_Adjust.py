import tensorflow as tf
from matplotlib import pyplot as plt

# a = tf.constant([1, 3, 4, 6])
img_path = r'../jiejie.jpg'

img_raw = tf.gfile.FastGFile(img_path, 'rb').read()
# print(img_raw)
img = tf.image.decode_jpeg(img_raw)
# print(img)




with tf.Session() as sess:
    img = img.eval()
    img = tf.cast(img, tf.float32)
    print(img.dtype, img.shape)
    a = tf.constant([[[[1, 2]]]], dtype=tf.float32)

    # contrast_factor = tf.constant([[[[3,4]]]],dtype=tf.float32)
    # tf.image.adjust_contrast()
    # 调整对比度
    b = tf.raw_ops.AdjustContrastv2(images=img, contrast_factor=0.5)
    # 调整色相
    c = tf.raw_ops.AdjustHue(images=img, delta=0.5)
    # 调整饱和度
    d = tf.raw_ops.AdjustSaturation(images=img, scale=0.5)
    deltas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]



    for i in deltas:
        # 调整饱和度
        d = tf.raw_ops.AdjustSaturation(images=img, scale=i)

    # print(sess.run(b))
    # print(sess.run(c))
    #     b = tf.cast(b, tf.uint8)
    #     c = tf.cast(c, tf.uint8)
        d=tf.cast(d,tf.uint8)
        img_ = d.eval()
        print(img_.shape)
        plt.figure(1)
        plt.imshow(img_)
        plt.savefig('../pic_save/'+str(i)+'.jpg')
        plt.show()
