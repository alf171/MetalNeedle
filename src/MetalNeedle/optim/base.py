from typing import NoReturn, List

from MetalNeedle import Tensor


class Optimizer:
    """" Base class for all optimizers """

    def __init__(self, parameters):
        self.parameters = parameters

    def zero_grad(self):
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad._zero()

    def step(self) -> NoReturn:
        raise NotImplemented("step not implemented")