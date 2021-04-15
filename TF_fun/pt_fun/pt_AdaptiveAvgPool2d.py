import torch
"""
输入tensor的size需大于2个维度
"""
a=torch.randn([1,3,3])
print(a.size())
a_=torch.tensor([[1,2,4,6],
                 [4,5,6,7]],dtype=torch.float)
print(a_.size())
b=torch.nn.AdaptiveAvgPool2d(output_size=[2,2])
c=b(a)
print(c.size())

