import torch
import torch.nn as nn


class RaIN(nn.Module):
    # ETTh2 24 eps=1e-9
    def __init__(self, num_features: int, eps=1e-5, max_min=True):

        super().__init__()
        self.num_features = num_features
        self.eps = eps

        self.max_min = max_min


    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError('Only modes norm and denorm are supported.')
        return x
    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))

        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        # self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

        # If the parameter for unbiased is changed to True, the sample standard deviation is calculated instead, and False is the overall standard deviation
        self.stdev = torch.std(x, dim=dim2reduce, keepdim=True, unbiased=True).detach() + self.eps  # 添加小的正数

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        # max_min scaling will scale the data to between [0-1] and may need to be adjusted to add a number to the denominator depending on the dataset
        if self.max_min:
            self.x_min = x.min()
            self.x_max = x.max()
            x = (x - self.x_min) / (self.x_max - self.x_min)
        return x

    def _denormalize(self, x):

        # The inverse operation of maximizing and minimizing needs to be decided on a task-specific basis
        #x = x * (self.x_max - self.x_min) + self.x_min
        x = x * self.stdev
        x = x + self.mean

        return x
