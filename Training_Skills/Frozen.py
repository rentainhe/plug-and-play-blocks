import torch
import torch.nn as nn
import torch.nn.functional as F


# You should put it in the Net

def frozen(module):
    if getattr(module, 'module', False):
        for child in module.module():
            for param in child.parameters():
                param.requires_grad = False
    else:
        for param in module.parameters():
            param.requires_grad = False