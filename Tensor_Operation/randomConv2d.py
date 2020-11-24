import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Conv2d
from torch.nn.modules.utils import _single, _pair, _triple, _repeat_tuple
import math

# def myconv2d(input, weight, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1):
#     # pad image and get parameter sizes
#     input = F.pad(input=input, pad= [padding[0], padding[0], padding[1], padding[1]], mode='constant', value=0)
#     dh, dw = stride
#     out_channels, in_channels, kh, kw = weight.shape
#     batch_size = input.shape[0]
#
#     # unfold input
#     patches = input.unfold(2, kh, dh).unfold(3, kw, dw)
#     h_windows = patches.shape[2]
#     w_windows = patches.shape[3]
#     patches = patches.expand(out_channels, *patches.shape)
#     patches = patches.permute(1, 3, 4, 0, 2, 5, 6)
#     patches = patches.contiguous()
#     # print(patches.shape)
#     # > torch.Size([batch_size, h_windows, w_windows, out_channels, in_channels, kh, kw])
#
#     # use our filter and sum over the channels
#     patches = patches * weight
#     patches = patches.sum(-1).sum(-1).sum(-1)
#
#     # add bias
#     if bias is not None:
#         bias = bias.expand(batch_size, h_windows, w_windows, out_channels)
#         patches = patches + bias
#     patches = patches.permute(0, 3, 1, 2)
#     # print(patches.shape)
#     # > torch.Size([batch_size, out_channels, h_windows ,w_windows])
#     return patches

# def myconv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
#     batch_size, in_channels, in_h, in_w = input.shape
#     out_channels, in_channels, kh, kw =  weight.shape
#
#     unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=dilation, padding=padding, stride=stride)
#     inp_unf = unfold(input)
#
#     if bias is None:
#         out_unf = inp_unf.transpose(1, 2).matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)
#     else:
#         out_unf = (inp_unf.transpose(1, 2).matmul(w_) + bias).transpose(1, 2)
#     out = out_unf.view(batch_size, out_channels, out_h, out_w)
#     return out

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

class RandomConv2d(nn.Module):
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
        mask = torch.ones((patch_nums,self.kernel_size[0]*self.kernel_size[1]),device=x.device) # 2,2
        index = torch.cat([torch.arange(index.shape[0]).unsqueeze(1),index],dim=-1)
        mask = mask.index_put((index[:,0],index[:,1]),torch.zeros(1)) # (9,4)
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
        out_unf = inp_unf.transpose(1,2).unsqueeze(2).matmul(masked_weight).squeeze().transpose(1,2)
        out = out_unf.view(b,self.out_channels,out_h,out_w)
        return out


r = RandomConv2d(3,16,3) # kernel_size = (2,2)
t = torch.randn(2,3,224,224) # feature_map = (4,4) 应该有9个卷积核
print(r(t).size())