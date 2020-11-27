import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Conv2d
from torch.nn.modules.utils import _single, _pair, _triple, _repeat_tuple
import math

'''
    使用unfold和fold实现了Conv2d操作
    groups参数有问题
'''
class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True):
        super(Conv2d, self).__init__()
        self.nums = 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.dilation = dilation
        self.stride = _pair(stride)
        self.padding = padding
        self.groups = groups
        self.unfold = nn.Unfold(self.kernel_size, self.dilation, self.padding, self.stride)
        self.weight = Parameter(torch.Tensor(out_channels,in_channels//groups,*self.kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self,x):
        b,c,h,w = x.size()
        out_h = (h - self.kernel_size[0] + self.padding*2) // self.stride[0] + 1 # 下一层 feature map的 h
        out_w = (w - self.kernel_size[1] + self.padding*2) // self.stride[1] + 1 # 下一层 feature map的 w
        inp_unf = self.unfold(x) # b, c*kh*kw, patch_nums
        weight = self.weight.view(self.weight.size(0), -1).t()
        out_unf = inp_unf.transpose(1,2).matmul(weight).transpose(1,2)
        out = out_unf.view(b,self.out_channels,out_h,out_w)
        return out

c = Conv2d(3,16,2)
r = torch.randn(1,3,4,4)
print(c(r).size())

