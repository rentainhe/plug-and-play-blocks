import torch.nn.functional as F
import torch
import torch.nn as nn


def PaddingV1(input, height, width):
    input_height = input.size(2)
    input_width = input.size(3)
    output = input
    if input_height < height:
        padding_height = height - input_height
        padding_upside = padding_height // 2
        padding_downside = padding_height - padding_upside
        output = F.pad(output, (0, 0, padding_upside, padding_downside), mode="constant", value=0)
    else:
        output = output[:, :, :height, :]
    if input_width < width:
        padding_width = width - input_width
        padding_leftside = padding_width // 2
        padding_rightside = padding_width - padding_leftside
        output = F.pad(output, (padding_leftside, padding_rightside, 0, 0), mode="constant", value=0)
    else:
        output = output[:, :, :, :width]
    return output


def PaddingV2(input, height, width):
    input_height = input.size(2)
    input_width = input.size(3)
    output = input
    if input_height < height:
        padding_height = height - input_height
        output = F.pad(output, (0, 0, 0, padding_height), mode="constant", value=0)
    else:
        output = output[:, :, :height, :]
    if input_width < width:
        padding_width = width - input_width
        output = F.pad(output, (0, padding_width, 0, 0), mode="constant", value=0)
    else:
        output = output[:, :, :, :width]
    return output