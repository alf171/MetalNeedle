from ..module import Module

class Sigmoid(Module):
    """
    softmax(x_i) = e^(x_i) / sum_{j}(e^x_j)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.max(0)
