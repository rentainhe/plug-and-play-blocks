import torch
import torch_extras
# expand_along(var, mask)
# Useful for selecting a dynamic amount of items from different indexes using a byte mask
setattr(torch, 'expand_along', torch_extras.expand_along)

var = torch.Tensor([1,0,2])
# var是需要index select的数据
# mask是select的规则
# 除了需要选择的维度，其余维度上mask必须和var一致
mask = torch.ByteTensor([[True,True],[False,True],[False,False]])
# 输出的结果是一维的
print(torch.expand_along(var,mask)) # 1,1,0

