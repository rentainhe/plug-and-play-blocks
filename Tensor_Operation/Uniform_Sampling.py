import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
    按kernel均匀采样
    index表示采样的位置，index=[1,0] 表示对于 kernel_size=1*2或者2*1大小的filter，采样左或者上位置
                       index=[1,0,0,0] 表示对于 kernel_size=2*2 的filter，采样左上角
                       index=[0,1,0,0] 表示对于 kernel_size=2*2 的filter，采样右上角
                       index=[1,0,0,0,0,0,0,0,0] 表示对于 kernel_size=3,3 的filter, 采样右上角
'''
class uniform_sampling(nn.Module):
    def __init__(self, kernel_size=(2, 2), index=[1, 0, 0, 0], padding=0):
        super(uniform_sampling, self).__init__()
        self.kernel_size = kernel_size  # (2,2)
        self.kh, self.kw = self.kernel_size[0], self.kernel_size[1] # kernel_h, kernel_w
        self.stride = self.kernel_size  # (2,2)
        self.index = index  # (2,2)
        assert len(index) == self.kh*self.kw, 'index必须可以被reshape为kernel_size大小'
        self.padding = padding  # default = None

    def forward(self, x):
        b, c, h, w = x.size()
        index = torch.from_numpy(np.array(self.index))
        index = index.view(self.kernel_size)  # 2,2
        index = index.unsqueeze(0).unsqueeze(0)  # 1,1,2,2
        x = F.conv2d(x, index.repeat(c, 1, 1, 1).float(), stride=self.stride, padding=self.padding, groups=c)
        return x