from ..module import Module
from ... import Tensor


class ReLU(Module):
    """
    y = max(x, 0)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor):
        return x.maximum(0)