import tensorflow as tf

a = tf.constant([1, -2, 6, -9])
y = tf.constant([2, 3, 6, 5])
a_ = tf.constant([1], dtype=tf.float32)
Abs = tf.raw_ops.Abs(x=a)
Acos = tf.raw_ops.Acos(x=a_)
Acosh = tf.raw_ops.Acosh(x=a_)
Add_N = tf.raw_ops.AddN(inputs=[[7,8,4,7],a])
Add_v2 = tf.raw_ops.AddV2(x=a, y=y)
input_x=tf.constant([2,4,6,9],dtype=tf.float32)
input_y=tf.constant([2,9,10,8],dtype=tf.float32)
Asin=tf.raw_ops.Asin(x=input_x)
Asinh=tf.raw_ops.Asinh(x=input_x)
Atan=tf.raw_ops.Atan(x=input_x)
Atan2=tf.raw_ops.Atan2(x=input_x,y=input_y)
Atanh=tf.raw_ops.Atanh(x=input_x)
Cos=tf.raw_ops.Cos(x=input_x)
Cosh=tf.raw_ops.Cosh(x=input_x)
ten_list=[Abs,Acos,Acosh,Add_N,Add_v2,Asin,Asinh,Atan,Atan2,Atanh,Cos,Cosh]

with tf.Session() as sess:
    for i in ten_list:
        # print(i)
        print('{}'.format(i.name),sess.run(i))

