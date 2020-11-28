import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from Hooks.pruning_network.model import LeNet

'''
    需要继承prune.BasePruningMethod方法
'''
class myPruningClass(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        print(default_mask.size())
        mask = default_mask.clone()
        mask.view(-1)[::2] = 0
        return mask

'''
    需要定义一个函数来使用这个类
'''
def myPruningMethod(module, name):
    myPruningClass.apply(module,name)
    return module

model = LeNet()
myPruningMethod(model.conv1, name='weight')
print(model.conv1.weight_mask)