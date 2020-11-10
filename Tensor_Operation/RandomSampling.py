import torch.nn as nn
import torch
import torch.nn.functional as F

class DistanceSampling(nn.Module):
    def __init__(self, num=1, kernel_size=(2, 2), stride=2, p=2, mode='Pairwise'):
        super(DistanceSampling, self).__init__()
        self.num = num
        self.stride = stride
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, stride=stride)
        self.mode = mode
        self.p = p
        self.mode = mode

    def get_ind(self, x, n):
        '''
            x.size: nums,hidden_size
        '''
        nums, c = x.size()
        mean_vector = torch.mean(x, dim=0, keepdim=True)  # 1,c
        mean_vector = mean_vector.repeat(nums, 1)  # nums,c
        distance = nn.PairwiseDistance(p=n, keepdim=True)
        Ln_distance = distance(x, mean_vector)
        values, index = torch.max(Ln_distance, dim=0)
        return index

    def forward(self, x):
        '''
        :param x:  b,c,h,w
        :param mask: b,1,h,w
        :return:
        '''
        b, c, h, w = x.size()  # 传入数据的维度
        feature_h = (h - self.kernel_size[0]) // self.stride + 1
        feature_w = (w - self.kernel_size[1]) // self.stride + 1
        output = self.unfold(x).view(b, c, self.kernel_size[0] * self.kernel_size[1],
                                     ((h - self.kernel_size[0]) // self.stride + 1) * ((w - self.kernel_size[
                                         1]) // self.stride + 1))  # b,c,kw*kh,w*h//(stride**2)
        x = self.unfold(x)
        num = x.size()[2]
        x = x.transpose(1, 2).contiguous().view(1, num, c, -1).transpose(2, 3)  # 1,kernel_nums,kernel_size,hidden_size
        x = x.squeeze()  # kernel_nums, kernel_size, hidden_size
        index = self.get_ind(x[0], self.p)
        for i in range(num - 1):
            index = torch.cat((index, self.get_ind(x[i + 1], self.p)), dim=0)
        index = index.long()
        index = index.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(1, c, 1, 1)
        output = output.gather(2, index)
        output = output.transpose(2, 3).contiguous().view(1, c, feature_h * feature_w)
        return output


def Transpose_tensor(input_tensor):
    '''
        input: b,c,h,w
        if h > w:
            output: b,c,w,h
        else:
            output: b,c,h,w
    '''
    b, c, h, w = input_tensor.size()
    if (h > w):
        input_tensor = input_tensor.transpose(2, 3)
    return input_tensor


def Filter(input_tensor):
    '''
        input: b,c,nums
        output: b,c,new_nums 把0的部分过滤
    '''
    b, c, nums = input_tensor.size()
    mask = torch.sum(input_tensor, 1) != 0
    T = (mask == 1)
    n = sum(T.squeeze() == 1)
    output = torch.masked_select(input_tensor, mask.repeat(b, c, 1))
    output = output.view(b, c, -1)
    return input_tensor, int(n)

class TotalRandomSamplingV2(nn.Module):
    def __init__(self, ratio=0.5):
        # input must be 3-d (b,c,feature_nums)
        super(TotalRandomSamplingV2, self).__init__()
        self.ratio = ratio
    def forward(self, x):
        b, c, nums = x.size()  # 传入数据的维度
        total_index = nums
        index_num = int(nums*self.ratio)
        probs = torch.ones(b,total_index)
        index = torch.multinomial(probs,index_num).unsqueeze(1)
        x = x.gather(2,index.repeat(1,c,1))
        return x

class TotalRandomSampling(nn.Module):
    '''
        input: b,c,h,w
        完全随机抽取特征
        ratio表示降采样的倍数，feature map中的特征数量必须能被ratio整除
        ratio=2表示随机抽取一半特征，降采样倍数=2
        output: b,c,nums ( if ratio=2, nums should be h*w//2 )
    '''
    def __init__(self, ratio=2):
        # input must be 3-d (1,c,feature_nums)
        super(TotalRandomSampling, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        b, c, nums = x.size()  # 传入数据的维度
        total_index = nums
        index_num = nums // self.ratio
        probs = torch.ones(total_index)
        index = torch.multinomial(probs, index_num).unsqueeze(0).unsqueeze(0)
        x = x.gather(2, index.repeat(1, c, 1))
        return x


class SparseRandomSampling(nn.Module):
    def __init__(self, num=1, kernel_size=(4, 4), stride=4):
        super(SparseRandomSampling, self).__init__()
        self.nums = num
        self.stride = stride
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, stride=stride)

    def forward(self, x):
        '''
        :param x:  b,c,h,w
        :param mask: b,1,h,w
        :return: b,c,feature_h*feature_w*nums
        '''
        b, c, h, w = x.size()  # 传入数据的维度
        feature_h = (h - self.kernel_size[0]) // self.stride + 1
        feature_w = (w - self.kernel_size[1]) // self.stride + 1
        temp_size = self.unfold(x).size()
        x = self.unfold(x).view(b, c, self.kernel_size[0] * self.kernel_size[1],
                                ((h - self.kernel_size[0]) // self.stride + 1) * (
                                            (w - self.kernel_size[1]) // self.stride + 1))
        hot = torch.zeros(x.size())
        probs = torch.ones([b, self.kernel_size[0] * self.kernel_size[1]], device=x.device, dtype=x.dtype)
        sample_ind = torch.cat([torch.multinomial(p, self.nums).unsqueeze(0) for p in probs], 0).unsqueeze(1).unsqueeze(-1)
        for i in range(x.size()[-1] - 1):
            temp = torch.cat([torch.multinomial(p, self.nums).unsqueeze(0) for p in probs], 0).unsqueeze(1).unsqueeze(
                -1)  # b,1,nums,1
            sample_ind = torch.cat((sample_ind, temp), 3)
        hot.scatter_(2, sample_ind.repeat(1, c, 1, 1), 1)
        hot = hot.view(temp_size)
        hot = F.fold(hot, output_size=(h, w), kernel_size=self.kernel_size, stride=self.stride)
        x = x.gather(2, sample_ind.repeat(1, c, 1, 1))
        x = x.transpose(2, 3).contiguous().view(1, c, feature_h * feature_w*self.nums)
        return x


class TotalRandom2D(nn.Module):
    def __init__(self, num=1, kernel_size=(2, 2), stride=2):
        super(TotalRandom2D, self).__init__()
        self.nums = num
        self.stride = stride
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, stride=stride)

    #         padding=(kernel_size-1)//2
    def forward(self, x):
        '''
        :param x:  b,c,h,w
        :param mask: b,1,h,w
        :return:
        '''
        b, c, h, w = x.size()  # 传入数据的维度

        x = self.unfold(x).view(b, c, self.kernel_size[0] * self.kernel_size[1],
                                ((h - self.kernel_size[0]) // self.stride + 1) * (
                                            (w - self.kernel_size[1]) // self.stride + 1))  # b,c,kw*kh,w*h//(stride**2)
        # 将数据重新组织，先使用self.unfold(x)将数据按块组织，然后再使用view函数，重新组织成四维，前两个维度和batch，channel一致
        #         mask=self.unfold(mask).view(b,1,self.kernel_size**2,h*w//(self.stride**2))
        probs = torch.ones([b, self.kernel_size[0] * self.kernel_size[1]], device=x.device, dtype=x.dtype)
        sample_ind = torch.cat([torch.multinomial(p, self.nums).unsqueeze(0) for p in probs], 0).unsqueeze(1).unsqueeze(
            -1)  # b,1,nums,1
        for i in range(x.size()[-1] - 1):
            temp = torch.cat([torch.multinomial(p, self.nums).unsqueeze(0) for p in probs], 0).unsqueeze(1).unsqueeze(
                -1)  # b,1,nums,1
            sample_ind = torch.cat((sample_ind, temp), 3)
        #         mask=mask.gather(2,sample_ind)
        x = x.gather(2, sample_ind.repeat(1, c, 1, 1))
        x = x.view(b, c, (h - self.kernel_size[0]) // self.stride + 1, (w - self.kernel_size[1]) // self.stride + 1)
        return x


class RandomPooling2D(nn.Module):
    def __init__(self, num=2, kernel_size=(2, 2), stride=2, mode='avg'):

        super(RandomPooling2D, self).__init__()
        self.nums = num
        self.mode = mode
        self.stride = stride
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size, stride=stride)

    #         padding=(kernel_size-1)//2
    def forward(self, x):
        '''
        :param x:  b,c,h,w
        :param mask: b,1,h,w
        :return:
        '''
        b, c, h, w = x.size()  # 传入数据的维度

        x = self.unfold(x).view(b, c, self.kernel_size[0] * self.kernel_size[1],
                                ((h - self.kernel_size[0]) // self.stride + 1) * (
                                            (w - self.kernel_size[1]) // self.stride + 1))
        probs = torch.ones([b,self.kernel_size[0]*self.kernel_size[1]], device=x.device, dtype=x.dtype)
        sample_ind = torch.cat([torch.multinomial(p, self.nums).unsqueeze(0) for p in probs], 0).unsqueeze(1).unsqueeze(-1)  # b,1,nums,1
        sample_ind = sample_ind.repeat(1, 1, 1, x.size()[-1])
        x = x.gather(2, sample_ind.repeat(1, c, 1, 1))
        if (self.mode == 'max'):
            x = torch.max(x, 2)[0]  # max会返回两个值，一个是max后的tensor，一个是索引
            x = x.view(b, c, (h - self.kernel_size[0]) // self.stride + 1, (w - self.kernel_size[1]) // self.stride + 1)
        elif (self.mode == 'avg'):
            x = torch.mean(x, 2)[0]
            x = x.view(b, c, (h - self.kernel_size[0]) // self.stride + 1, (w - self.kernel_size[1]) // self.stride + 1)
        elif (self.mode == 'weighted'):
            scores = F.softmax(x, dim=2)
            x = torch.sum(torch.mul(x, scores), dim=2)
            x = x.view(b, c, (h - self.kernel_size[0]) // self.stride + 1, (w - self.kernel_size[1]) // self.stride + 1)
        return x



