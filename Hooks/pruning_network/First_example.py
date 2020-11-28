import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from Hooks.pruning_network.model import LeNet

# Create a model for test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device=device)

module = model.conv1
# print(list(module.named_parameters())) # only contain 'weight' and 'bias'
# print(list(module.named_buffers())) # no buffers

'''
    First example
'''
# select a pruning technique or design your own pruning method
prune.random_unstructured(module, name='weight', amount=0.3) # 给权重加一个mask

# this method will random prune 30% of the connections in the parameter named weight in the conv1 layer
# name identifies the parameter within that module using its string identifier
# amount 表示 pruning 的比率
print(list(module.named_parameters()))
print(list(module.named_buffers()))
print(module.weight) # 会随机mask掉30%的weight