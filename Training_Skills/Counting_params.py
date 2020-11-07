import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    counting the params whose requires_grad is True
'''
# Version 1
def count_parameters(net):
    params = sum([param.nelement() for param in net.parameters() if param.requires_grad])
    print("Params: %f M" % (params/1000000))

# Version 2
def print_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')