from torch.autograd import Function
import torch.nn as nn
import torch
import torch.nn.functional as F

# class my_function(Function):
#     def forward(self, input, parameters):
#         self.saved_for_backward = [input, parameters]
#         # output = [do something with input and parameters]
#         return output
#
#     def backward(self, grad_output):
#         input, parameters = self.saved_for_backward
#         # grad_input = [derivate forward(input) wrt parameters] * grad_output
#         return grad_input
#
#
# class my_module(nn.Module):
#     def __init__(self, ...):
#         super(my_module, self).__init__()
#         self.parameters = # init some parameters
#
#     def backward(self, input):
#         output = my_function(input, self.parameters) # here you call the function!
#         return output
