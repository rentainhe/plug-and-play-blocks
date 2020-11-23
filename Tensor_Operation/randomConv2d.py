import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Conv2d
from torch.nn.modules.utils import _single, _pair, _triple, _repeat_tuple
import math
# class randomConv2d(Conv2d):
#     def __init__(self,in_channels, out_channels, kernel_size, nums,stride=1,padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros'):
#         super(randomConv2d,self).__init__()
#         self.nums = nums # 采样的数量
#         self.unfold = nn.Unfold(kernel_size,dilation,padding,stride)
#
#     def forward(self, input):
#         print(self.weight.size())
#         out_channels, in_channels_divided, kh, kw = self.weight.size()
#         probs = torch.ones(kh*kw)
#         index = torch.multinomial(probs,self.nums)
#         self.mask[index] = 0
#         mask = self.mask.view(kh,kw).unsqueeze(0).unsqueeze(0).repeat(out_channels,1,1,1)
#         self.weight = torch.mul(self.weight,mask)
#         return self._conv_forward(input,self.weight)


class randomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, nums,stride=1,padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros'):
        super(randomConv2d,self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.nums = nums
        self.unfold = nn.Unfold(kernel_size,dilation,padding,stride)
        self.weight = Parameter(torch.Tensor(out_channels,in_channels//groups,*kernel_size))
        self.mask = Parameter(torch.ones(kernel_size[0]*kernel_size[1]),requires_grad=False)
        self.padding_mode = padding_mode
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self._padding_repeated_twice = _repeat_tuple(self.padding, 2)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters( )

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _conv_forward(self,input,weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode,),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input,weight,self.bias,self.stride,
                        self.padding,self.dilation,self.groups)

    def forward(self, input):
        # input = self.unfold(input)
        print(self.weight.size())
        out_channels, in_channels_divided, kh, kw = self.weight.size()
        probs = torch.ones(kh*kw)
        index = torch.multinomial(probs,self.nums)
        # 不能对parameter中的数值直接赋值，比如 self.conv = self.conv 啥的，必须要让weight=self.weight，再对weight进行操作
        self.mask[index] = 0
        weight = torch.mul(self.weight,self.mask.view(kh,kw).unsqueeze(0).unsqueeze(0).repeat(out_channels,1,1,1))
        print(weight)
        return self._conv_forward(input, weight)

# r = randomConv2d(4,16,2,1,2,groups=4)
# t = torch.randn(1,4,4,4)
# print(r(t).size())

class myConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True):
        super(myConv, self).__init__()
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
        out_h = (h - self.kernel_size[0] + self.padding*2) // self.stride[0] + 1
        out_w = (w - self.kernel_size[1] + self.padding*2) // self.stride[1] + 1
        inp_unf = self.unfold(x)
        out_unf = inp_unf.transpose(1,2).matmul(self.weight.view(self.weight.size(0), -1).t()).transpose(1,2)
        out = out_unf.view(b,self.out_channels,out_h,out_w)
        return out

r = myConv(3,16,2,1)
t = torch.randn(2,3,4,4)
print(r(t).size())