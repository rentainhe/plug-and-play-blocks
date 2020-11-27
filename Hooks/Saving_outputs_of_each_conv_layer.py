import torch
import torch.nn as nn
from torchvision.models import resnet34

'''
    you can register a hook as a 
    1. forward prehook (executing before the forward pass)
    2. forward hook (executing after the forward pass)
    3. backward hook (executing after the backward pass)
'''

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = resnet34(pretrained=False)
model = model.to(device)

# the basic callable object
class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self,module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

save_output = SaveOutput()

hook_handles = []

for layer in model.modules():
    if isinstance(layer, nn.modules.conv.Conv2d): # check if the layer is a Conv layer
        handle = layer.register_forward_hook(save_output)
        hook_handles.append(handle)
# after this is done, the hook will be called after each forward pass of each convolutional layer

test = torch.randn(1,3,224,224)
out = model(test)
print(save_output.outputs[0][0].size()) # 中间层结果的打印
