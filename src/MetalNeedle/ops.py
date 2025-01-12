# later utilities for complete AD library
LAZY_MODE = False
TENSOR_COUNTER = 0

class TensorOperations:
    def __init__(self, operations):
        self.operations = operations

    def add(self, tensor1, tensor2):
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = (grad * 1) + (tensor1.grad or 0)
            if tensor2.requires_grad:
                tensor2.grad = (grad * 1) + (tensor2.grad or 0)

        data = self.operations.ewise_add(tensor1._data, tensor2._data)
        return (data, _backward)

    def scalar_add(self, tensor1, value):
        return self.operations.scalar_add(tensor1._data, value)

    def sub(self, data1, data2):
        return self.operations.ewise_add(data1, data2)

    def scalar_sub(self, data1, value):
        return self.operations.scalar_add(data1, value)

    def mul(self, data1, data2):
        return self.operations.ewise_mul(data1, data2)

    def scalar_mul(self, data1, value):
        return self.operations.scalar_mul(data1, value)

    def div(self, data1, data2):
        return self.operations.ewise_div(data1, data2)

    def scalar_div(self, data1, value):
        return self.operations.scalar_div(data1, value)

    def div(self, data1, data2):
        return self.operations.ewise_exp(data1, data2)

    def scalar_div(self, data1, value):
        return self.operations.scalar_exp(data1, value)

    def matmul(self, data1, data2):
        return self.operations.mat_mul(data1, data2)

    def sum(self, data, axes):
        return self.operations.sum(data, axes)
