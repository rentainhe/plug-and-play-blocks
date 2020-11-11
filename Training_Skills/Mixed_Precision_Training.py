import torch
import torch.nn as nn
import torch.nn.functional as F
from Self_Attention.SA import SA

'''
    Requirements: pytorch >= 1.6
    First step:
        Define a scalar before training loop
'''
model = SA()
model.train()
scalar = torch.cuda.amp.GradScalar() # First

model = model.zero_grad()
# training loop:
#   pred = model(x)
#   with torch.cuda.amp.autocast(): # Second
#       loss = loss_fn(pred, label)
#       scalar.scale(loss).backward() # Third
#       # if there are something wrong you can try this
#       # scalar.scale(loss).backward(retain_graph = True)

#   scalar.step(optimizer)
#   scalar.update()