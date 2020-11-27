import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Conv2d
from torch.nn.modules.utils import _single, _pair, _triple, _repeat_tuple
import math

class RandomConv2d(nn.Module):
    # useless
    def __init__(self, in_channels, out_channels, kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True):
        super(RandomConv2d, self).__init__()
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

    def get_masked_weight(self,x,weight,patch_nums):
        # 12，16 = 3*2*2，16
        # 原来的weight是12*16，需要得到9个不同的weight，所以需要产生9组不同的12*16的weight，再拼接
        b, in_channels,h,w = x.size()
        weight_nums, out_channels = weight.size() # 一个Filter的weight数量，输出的channel数
        probs = torch.ones(patch_nums, self.kernel_size[0]*self.kernel_size[1]) # 9,4
        index = torch.multinomial(probs,self.nums)
        # index = index.cuda()
        mask = torch.ones((patch_nums,self.kernel_size[0]*self.kernel_size[1]),device=x.device) # 2,2
        index = torch.cat([torch.arange(index.shape[0],device=x.device).unsqueeze(1),index],dim=-1)
        mask = mask.index_put((index[:,0],index[:,1]),torch.zeros(1,device=x.device)) # (9,4)
        mask = mask.repeat(1,in_channels).view(mask.size()[0],-1) # 9,12
        weight = weight.unsqueeze(0).repeat(patch_nums,1,1)
        return (weight*mask.unsqueeze(2)) # 9,12,16



    def forward(self,x):
        b,c,h,w = x.size()
        out_h = (h - self.kernel_size[0] + self.padding*2) // self.stride[0] + 1
        out_w = (w - self.kernel_size[1] + self.padding*2) // self.stride[1] + 1
        inp_unf = self.unfold(x) # b, c*kh*kw, patch_nums
        # 2,9,1,12
        # 9,12,16
        weight = self.weight.view(self.weight.size(0), -1).t()
        patch_nums = inp_unf.size()[2] # 2,12,9
        # out_unf = inp_unf.transpose(1,2).matmul(weight).transpose(1,2)
        masked_weight = self.get_masked_weight(x,weight,patch_nums)
        # print(masked_weight)
        out_unf = inp_unf.transpose(1,2).unsqueeze(2).matmul(masked_weight).flatten(2).transpose(1, 2)
        out = out_unf.view(b,self.out_channels,out_h,out_w)
        return out





