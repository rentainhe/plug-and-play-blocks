import torch
import torch.nn as nn
import torch.nn.functional as F
# 通过index将tensor分成两部分，一部分是选中的，一部分是未选中的

'''
    x: b,c,h,w
    mask: 一维，长度和需要进行选择的维度一致
    dim: 需要进行index选择的维度
'''
def index_select(x,mask,dim):
    b, c, h, w = x.size()
    index = torch.zeros(x.size()[dim]).bool()
    index[mask] = True
    select_index = index


x = torch.randn(2,1,4,4)
mask = [True,True]
