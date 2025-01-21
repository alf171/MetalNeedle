from .data import TensorData

LAZY_MODE = False
TENSOR_COUNTER = 0

class TensorOperations:
    def __init__(self, operations):
        # TODO: deprecate and make methods static
        self.operations = operations

    def add(self, tensor1, tensor2):
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = grad + (tensor1.grad or 0)
            if tensor2.requires_grad:
                tensor2.grad = grad + (tensor2.grad or 0)

        tensorData = tensor1.tensorData + tensor2.tensorData
        return (tensorData.rawTensor, _backward)

    def scalar_add(self, tensor1, value):
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = grad + (tensor1.grad or 0)

        tensorData = tensor1.tensorData + value
        return (tensorData.rawTensor, _backward)

    def sub(self, tensor1, tensor2):
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = grad + (tensor1.grad or 0)
            if tensor2.requires_grad:
                tensor2.grad = (-1 * grad) + (tensor2.grad or 0)

        tensorData  = tensor1.tensorData - tensor2.tensorData
        return (tensorData.rawTensor, _backward)

    def scalar_sub(self, tensor1, value):
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = grad + (tensor1.grad or 0)
        tensorData = tensor1.tensorData - value
        return (tensorData.rawTensor, _backward)

    def mul(self, tensor1, tensor2):
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = (tensor2.tensorData * grad) + (tensor1.grad or 0)
            if tensor2.requires_grad:
                tensor2.grad = (tensor1.tensorData * grad) + (tensor2.grad or 0)

        tensorData = tensor1.tensorData * tensor2.tensorData
        return (tensorData.rawTensor, _backward)

    def scalar_mul(self, tensor1, value):
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = (grad * value) + (tensor1.grad or 0)

        tensorData = tensor1.tensorData * value
        return (tensorData.rawTensor, _backward)

    def div(self, tensor1, tensor2):
        def _backward(grad):
            # da(A/B) = 1/B
            if tensor1.requires_grad:
                tensor1.grad = (grad / tensor2.tensorData) + (tensor1.grad or 0)
            # db(A/B) = -A/B^2
            if tensor2.requires_grad:
                tensor2.grad = (-grad * tensor1.tensorData) / (tensor2.tensorData ** 2) + (tensor1.grad or 0)

        tensorData = tensor1.tensorData / tensor2.tensorData
        return (tensorData.rawTensor, _backward)

    def scalar_div(self, tensor1, value):
        def _backward(grad):
            if tensor1.requires_grad:
                tensor1.grad = (grad / value) + (tensor1.grad or 0)
        tensorData = tensor1.tensorData / value
        return (tensorData.rawTensor, _backward)

    # TODO: add grad
    def exp(self, tensor1, tensor2):
        def _backward(grad):
            pass
        tensorData = tensor1.tensorData ** tensor2.tensorData
        return tensorData.rawTensor

    # TODO: add grad
    def scalar_exp(self, tensor1, value):
        def _backward(grad):
            pass
        tensorData = tensor1.tensorData ** value
        return tensorData.rawTensor

    # TODO: add grad (missing keep dim support I think?)
    def matmul(self, tensor1, tensor2):
        def _backward(grad):
            pass
        tensorData = tensor1.tensorData @ tensor2.tensorData
        return tensorData.rawTensor

    # TODO: add grad
    def sum(self, tensor1, axes):
        def _backward(grad):
            pass
        tensorData = tensor1.tensorData.sum(axes)
        return tensorData.rawTensor
