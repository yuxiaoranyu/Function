import tensorflow as tf

a=tf.constant([[2,4,5,4j],
               [6,2,5,4j]],dtype=tf.complex64)
# b=tf.raw_ops.IFFT(input=a)
c=tf.raw_ops.FFT(input=a)
# with tf.Session() as sess:
#     print(sess.run(b))
#     print(sess.run(c))

import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv=nn.Conv2d(3,3,4)
        nn.Sequential()

    def forward(self,a):
        out=self.conv(a)

        return out

a=torch.randn([3,3,4,8])
# b=torch.tensor([2,3])

# model=Net()
# x=model(a)
# print(model)
# torch.save(model,'conv.pt')
# print(x)

class add_Net(nn.Module):
    def __init__(self):
        super(add_Net, self).__init__()
        self.add=torch.add
        self.mv=torch.mv
    def forward(self,a,b):
        # res=self.add(a,b)
        # print(res)
        ten=torch.tensor([1,2,3,4])
        out=torch.matmul(a,b)
        # out=self.mv(res,ten)

        return out

mode=add_Net()
torch.save(mode,'add.pth')
# add_=torch.load('add.pth')
# print(add_)
a_=torch.tensor([[1,3],
                 [1,2]])

b_=torch.tensor([[7,6],
                 [1,2]])

ten=torch.tensor([1,2,3,4])
# res=torch.mv(a_,ten)
# print(res)
# x_=[]
# for i in a_:
#     print(i)
#     x=torch.matmul(i,ten)
#     x_.append(torch.tensor([x]))
#     print(x)

# m=torch.cat((x_[0]))
# print(m)
# a_t=torch.randn([1,2,3,4])
# b_t=torch.randn([1,2,3,4])
# res=mode(a_,b_)
# print(res)
torch.onnx.export(mode,(a_,b_),'add.onnx',verbose=False)