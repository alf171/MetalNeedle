from ..module import Module
from ... import Tensor

class Softmax(Module):
    """
    softmax(x_i) = e^(x_i) / sum_{j}(e^x_j)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x : Tensor):
        x_shifted = x - x.max(axes=-1, keep_dims=True).broadcast(x.shape())
        x_exp = x_shifted.exp()
        x_sum = x_exp.sum(axes=-1, keep_dims=True).broadcast(x.shape())
        return x_exp / x_sum
