# later utilities for complete AD library
from .data import TensorData
from .device import DeviceManager

LAZY_MODE = False
TENSOR_COUNTER = 0

class TensorOperations:
    def __init__(self, operations):
        self.operations = operations

    def add(self, tensor1, tensor2):
        t1 = TensorData.create(tensor1._data.tensor, self.operations)
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = grad + (tensor1.grad or 0)
            if tensor2.requires_grad:
                tensor2.grad = grad + (tensor2.grad or 0)

        data = t1 + tensor2._data.tensor
        return (data, _backward)

    def scalar_add(self, tensor1, value):
        t1 = TensorData.create(tensor1._data.tensor, self.operations)
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = grad + (tensor1.grad or 0)

        data = t1 + value
        return (data, _backward)

    def sub(self, tensor1, tensor2):
        t1 = TensorData.create(tensor1._data.tensor, self.operations)
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = grad + (tensor1.grad or 0)
            if tensor2.requires_grad:
                tensor2.grad = (-1 * grad) + (tensor2.grad or 0)

        data = t1 - tensor2._data.tensor
        return (data, _backward)

    def scalar_sub(self, tensor1, value):
        t1 = TensorData.create(tensor1._data.tensor, self.operations)
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = grad + (tensor1.grad or 0)
        data = t1 - value
        return (data, _backward)

    def mul(self, tensor1, tensor2):
        t1 = TensorData.create(tensor1._data.tensor, self.operations)
        t2 = TensorData.create(tensor2._data.tensor, self.operations)
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = TensorData.create((t2 * grad), self.operations) + (tensor1.grad or 0)
            if tensor2.requires_grad:
                tensor2.grad = TensorData.create((t1 * grad), self.operations) + (tensor2.grad or 0)

        data = t1 * tensor2._data.tensor
        return (data, _backward)

    def scalar_mul(self, tensor1, value):
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = (grad * value) + (tensor1.grad or 0)
        data = self.operations.scalar_mul(tensor1._data.tensor, value)
        return (data, _backward)

    # TODO: add grad
    def div(self, tensor1, tensor2):
        return self.operations.ewise_div(tensor1._data.tensor, tensor2._data.tensor)

    # TODO: add grad
    def scalar_div(self, tensor1, value):
        return self.operations.scalar_div(tensor1._data.tensor, value)

    # TODO: add grad
    def exp(self, tensor1, tensor2):
        return self.operations.ewise_exp(tensor1._data.tensor, tensor2._data.tensor)

    # TODO: add grad
    def scalar_exp(self, tensor1, value):
        return self.operations.scalar_exp(tensor1._data.tensor, value)

    # TODO: add grad
    def matmul(self, tensor1, tensor2):
        return self.operations.mat_mul(tensor1._data.tensor, tensor2._data.tensor)

    # TODO: add grad
    def sum(self, tensor1, axes):
        return self.operations.sum(tensor1._data.tensor, axes)
