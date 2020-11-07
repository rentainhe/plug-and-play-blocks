import torch
import torch.nn as nn
import torch.nn.functional as F

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000
