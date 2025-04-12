import math

from ..module import Module
from ... import Tensor


class Softmax(Module):
    """
    softmax(x_i) = e^(x_i) / sum_{j}(e^x_j)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x : Tensor):
        return x.exp() / x.sum(axes=-1, keepdim=True).broadcast(x.shape())
