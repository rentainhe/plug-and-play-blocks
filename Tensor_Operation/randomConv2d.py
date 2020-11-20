import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Conv2d

class randomConv2d(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, nums):
        super(randomConv2d,self).__init__()
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





