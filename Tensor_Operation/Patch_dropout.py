import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config():
    def __init__(self):
        self.PATCH_NUMS = 160
        self.PATCH_DROPOUT_R = 0.25


class Patch_Dropout(nn.Module):
    def __init__(self, __C, kernel_size=(2, 2), stride=(2, 2), padding=True, padding_size=(20, 32)):
        super(Patch_Dropout, self).__init__()
        self.__C = __C
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_size = padding_size
        assert padding_size[0] // kernel_size[0] == 0
        assert padding_size[1] // kernel_size[1] == 0

    def Padding(self, input, height, width):
        '''
        Padding operation
        :param input: b,c,h,w
        :param height: if h < height, padding h to height
        :param width:  if w < width, padding w to width
        :return: padded tensor
        '''
        input_height = input.size(2)
        input_width = input.size(3)
        output = input
        if input_height < height:
            padding_height = height - input_height
            output = F.pad(output, (0, 0, 0, padding_height), mode="constant", value=0)
        else:
            output = output[:, :, :height, :]
        if input_width < width:
            padding_width = width - input_width
            output = F.pad(output, (0, padding_width, 0, 0), mode="constant", value=0)
        else:
            output = output[:, :, :, :width]
        return output

    def Patch_Dropout(self, x, __C):
        b, c, k_size, patch_nums = x.size()
        patch_dropout_rate = __C.PATCH_DROPOUT_R
        assert patch_nums == __C.PATCH_NUMS
        undropped_nums = __C.PATCH_NUMS - int(patch_nums * patch_dropout_rate)
        probs = torch.ones((b, patch_nums))
        index = torch.multinomial(probs, undropped_nums)  # b,undropped_nums
        index = index.unsqueeze(1).unsqueeze(1).repeat(1, c, k_size, 1)  # b,1,k_size,undropped_nums
        # if you use GPU for training:
        # index = index.cuda() # GPU Version
        index = index # CPU Version
        x = x.gather(3, index)
        return x

    def forward(self, x):
        if self.padding:
            x = self.Padding(x, self.padding_size[0], self.padding_size[1])
        b, c, h, w = x.size()
        feats_in_one_kernel = self.kernel_size[0] * self.kernel_size[1]
        x = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        assert self.__C.PATCH_NUMS == h * w // feats_in_one_kernel
        x = x.view(b, c, feats_in_one_kernel, self.__C.PATCH_NUMS)  # b,c,4,160
        x = self.Patch_Dropout(x, self.__C)
        x = x.view(b, c, -1)
        x = x.transpose(1, 2)
        return x