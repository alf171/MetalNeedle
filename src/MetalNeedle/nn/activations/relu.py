from ..module import Module

class ReLU(Module):
    """
    y = max(x, 0)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.maximum(0)