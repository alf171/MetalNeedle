from typing import Any, NoReturn

class Optimizer:
    """" Base class for all optimizers """

    def __init__(self, parameters: list[Any]):
        self.parameters = parameters

    def zero_grad(self):
        for parameter in self.parameters:
            if parameter.grad is not None:
                parameter.grad = parameter.grad.zeros_like()

    def step(self) -> NoReturn:
        raise NotImplementedError
