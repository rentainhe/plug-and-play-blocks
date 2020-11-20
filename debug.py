import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Conv2d

class randomConv2d(Conv2d):
    def __init__(self,in_channels, out_channels, kernel_size, nums,stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups,
                 bias, padding_mode)
        self.nums = nums # 采样的数量
        self.kh = self.weight.size()[2]
        self.kw = self.weight.size()[3]
        self.mask = nn.Parameter(torch.ones(self.kh*self.kw),requires_grad=False)

    def forward(self, input):
        print(self.weight.size())
        out_channels, in_channels_divided, kh, kw = self.weight.size()
        probs = torch.ones(kh*kw)
        index = torch.multinomial(probs,self.nums)
        self.mask[index] = 0
        mask = self.mask.view(kh,kw).unsqueeze(0).unsqueeze(0).repeat(out_channels,1,1,1)
        self.weight = torch.mul(self.weight,mask)
        return self._conv_forward(input,self.weight)


r = randomConv2d(4,16,2,nums=1,groups=4)
t = torch.randn(1,4,4,4)
print(r(t).size())




