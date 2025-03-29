from src import MetalNeedle
from src.MetalNeedle.nn import Module


class Linear(Module):
    """
    y = xW^T + b
    where x: [batch_size, in_features]
    W: [out_features, in_features]
    b: [out_features]
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = MetalNeedle.randn([out_features, in_features])
        self.bias = MetalNeedle.randn([out_features])

    def forward(self, x):
        return x @ self.weight.T() + self.bias
