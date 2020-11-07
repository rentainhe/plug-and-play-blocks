import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    counting the params whose requires_grad is True
'''
def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000
