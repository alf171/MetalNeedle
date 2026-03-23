import math

from .module import Module
from MetalNeedle import randn, Tensor


class Linear(Module):
    """
    y = xW^T + b
    where x: [batch_size, in_features]
    W: [out_features, in_features]
    b: [out_features]
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        weight_std = math.sqrt(2.0 / (in_features + out_features))
        self.weight = randn([out_features, in_features], requires_grad=True) * weight_std
        self.weight.requires_grad = True
        self.bias = randn([out_features], mean=0, std=0, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return (x @ self.weight.T) + self.bias
