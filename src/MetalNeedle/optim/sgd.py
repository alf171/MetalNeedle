from distutils.sysconfig import parse_makefile

from MetalNeedle.optim.base import Optimizer


class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01, momentum=0, weight_decay=0):
        super().__init__(parameters)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # group, param = velocity
        self.velocity = {}

        if momentum > 0:
            for (i, param) in enumerate(self.parameters):
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

                self.velocity[i] = self.momentum * self.velocity[i] + grad
                update = self.velocity[i]
            else:
                update = grad


            print(f"old data sum: {param.sum()}")
            param.tensor_data -= (self.lr * update)
            print(f"new data sum: {param.sum()}")
