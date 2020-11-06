import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# If you have config file, you should add these variates
class Config():
    def __init__(self):
        self.FEATURE_NUMS = 640
        self.NORM_DROPOUT_R = 0.25
        self.KERNEL_SIZE = [2,2]
        self.DROPOUT_NUMS = 1

def get_kernel_nums(feature_nums, dropout_rate, kernel_size, dropout_nums):
    '''
    获得需要进行RandomSampling的kernel的数量
    :param feature_nums: 输入图像的feature map大小
    :param dropout_rate: dropout rate
    :param kernel_size:  kernel 大小
    :param dropout_nums: 一个kernel中需要dropout掉特征的数量
    :return: int
    '''
    kh, kw = kernel_size[0], kernel_size[1]
    max_dropout_rate = dropout_nums / (kh*kw)
    assert dropout_rate <= max_dropout_rate, 'the dropout rate you set is beyond the max dropout rate'
    num = int((feature_nums*dropout_rate) // (dropout_nums)) # 需要进行RandomSampling的kernel数量
    return num


class NormDropout(nn.Module):
    def __init__(self, num=2, kernel_size=(2, 2), stride=2, dropout_rate=0.1):
        super(NormDropout, self).__init__()
        self.nums = num
        self.dropout_nums = kernel_size[0]*kernel_size[1] - self.nums
        self.dropout_rate = dropout_rate
        self.stride = stride
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, stride=stride)

    def get_norm(self, x, num):
        b, c, h, w = x.size()
        output = x
        x = torch.mean(x, dim=2, keepdim=True)
        norm = torch.norm(x, dim=1, keepdim=True)
        index = torch.argsort(norm)
        index = index.squeeze()
        index1 = index[:num]
        index2 = index[num:]
        output1 = output[:, :, :, index1]
        output2 = output[:, :, :, index2]
        return output1, output2

    def random_drop(self, x, b, c):
        '''
            input b,c,4,160
        '''
        probs = torch.ones([1, self.kernel_size[0] * self.kernel_size[1]], device=x.device, dtype=x.dtype)
        sample_ind = torch.cat([torch.multinomial(p, self.nums).unsqueeze(0) for p in probs], 0).unsqueeze(1).unsqueeze(
            -1)  # b,1,nums,1
        for i in range(x.size()[-1] - 1):
            temp = torch.cat([torch.multinomial(p, self.nums).unsqueeze(0) for p in probs], 0).unsqueeze(1).unsqueeze(
                -1)  # b,1,nums,1
            sample_ind = torch.cat((sample_ind, temp), 3)
        x = x.gather(2, sample_ind.repeat(1, c, 1, 1))
        return x

    def forward(self, x):
        '''
        :param x:  b,c,h,w
        :param mask: b,1,h,w
        :return:
        '''
        b, c, h, w = x.size()  # 传入数据的维度
        feature_h = (h - self.kernel_size[0]) // self.stride + 1
        feature_w = (w - self.kernel_size[1]) // self.stride + 1
        x = self.unfold(x).view(b, c, self.kernel_size[0] * self.kernel_size[1],
                                ((h - self.kernel_size[0]) // self.stride + 1) * (
                                            (w - self.kernel_size[1]) // self.stride + 1))  # b,c,kw*kh,w*h//(stride**2)
        temp_1, temp_2 = self.get_norm(x, 20)
        temp_1 = self.random_drop(temp_1, b, c)
        temp_1 = temp_1.transpose(2, 3).contiguous().view(1, c, -1)
        temp_2 = temp_2.transpose(2, 3).contiguous().view(1, c, -1)
        output = torch.cat((temp_1, temp_2), 2)
        #         x = x.transpose(2,3).contiguous().view(1,c,feature_h*feature_w*3)
        return output
