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
    内置pruning，给权重结合一个mask
'''
# select a pruning technique or design your own pruning method
'''
    pruning weight
'''
prune.random_unstructured(module, name='weight', amount=0.3) # 给权重加一个mask
# this method will random prune 30% of the connections in the parameter named weight in the conv1 layer
# name identifies the parameter within that module using its string identifier
# amount 表示 pruning 的比率

'''
    pruning bias
'''
prune.l1_unstructured(module, name='bias', amount=3)

print(list(module.named_parameters()))
print(list(module.named_buffers())) # we got weight_mask and bias_mask here
print(module.weight) # 会随机mask掉30%的weight
print(module.bias)