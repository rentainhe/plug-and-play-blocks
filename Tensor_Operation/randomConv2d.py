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
        return self._conv_forward(input, self.weight)

r = randomConv2d(4,16,2,1,2,groups=4)
t = torch.randn(1,4,4,4)
print(r(t).size())
