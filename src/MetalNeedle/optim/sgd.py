from distutils.sysconfig import parse_makefile

from MetalNeedle.optim.base import Optimizer
from typing import Any


class SGD(Optimizer):
    def __init__(self, parameters: list[Any] , lr: float =0.01, momentum: float = 0, weight_decay: float =0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # group, param = velocity
        self.velocity = {}

        if momentum > 0:
            for i in range(len(self.parameters)):
                self.velocity[i] = None

    def step(self):
        for (i, param) in enumerate(self.parameters):
            if param.grad is None:
                print("skip!")
                continue

            grad = param.grad

            if self.weight_decay > 0:
                grad += self.weight_decay * param

            if self.momentum > 0:
                if self.velocity[i] is None:
                    self.velocity[i] = grad.zeros_like()

                self.velocity[i] = self.velocity[i] * self.momentum + grad
                update = self.velocity[i]
            else:
                update = grad

            param.tensor_data -= update * self.lr
