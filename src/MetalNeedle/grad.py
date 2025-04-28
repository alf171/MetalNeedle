from typing import TypeVar, Union, List, Any

from MetalNeedle.data import TensorData

T = TypeVar('T', bound=Union[int, float])

class TensorGrad:
    @staticmethod
    def add(grad: TensorData, tensor1, tensor2) -> None:
        if tensor1.requires_grad:
            tensor1.backward(grad)
        if tensor2.requires_grad:
            tensor2.backward(grad)

    @staticmethod
    def scalar_add(grad: TensorData, tensor1) -> None:
        if tensor1.requires_grad:
            tensor1.backward(grad)

    @staticmethod
    def sub(grad: TensorData, tensor1, tensor2) -> None:
        if tensor1.requires_grad:
            tensor1.backward(grad)
        if tensor2.requires_grad:
            tensor2_grad = -grad
            tensor2.backward(tensor2_grad)

    @staticmethod
    def scalar_sub(grad: TensorData, tensor1) -> None:
        if tensor1.requires_grad:
            tensor1.backward(grad)

    @staticmethod
    def mul(grad: TensorData, tensor1, tensor2) -> None:
        if tensor1.requires_grad:
            tensor1_grad = (tensor2.tensor_data * grad)
            tensor1.backward(tensor1_grad)
        if tensor2.requires_grad:
            tensor2_grad = (tensor1.tensor_data * grad)
            tensor2.backward(tensor2_grad)

    @staticmethod
    def scalar_mul(grad: TensorData, tensor1, value: T) -> None:
        if tensor1.requires_grad:
            tensor1.grad = (grad * value)

    @staticmethod
    def div(grad: TensorData, tensor1, tensor2) -> None:
        # da(A/B) = 1/B
        if tensor1.requires_grad:
            tensor1_grad = (grad / tensor2.tensor_data)
            tensor1.backward(tensor1_grad)
        # db(A/B) = -A/B^2
        if tensor2.requires_grad:
            tensor2_grad = (-grad * tensor1.tensor_data) / (tensor2.tensor_data ** 2)
            tensor2.backward(tensor2_grad)

    @staticmethod
    def scalar_div(grad: TensorData, tensor1, value: T) -> None:
        if tensor1.requires_grad:
            tensor1.backward(grad / value)

    @staticmethod
    def pow(grad: TensorData, tensor1, tensor2) -> None:
        # dx(x^y) = y * x^(y-1)
        if tensor1.requires_grad:
            tensor1_grad = (grad * tensor2.tensor_data * (tensor1.tensor_data ** (tensor2.tensor_data - 1)))
            tensor1.backward(tensor1_grad)
        # dy(x^y) = dy(e^(y*lnx)) = lnx*e^(y*lnx) = lnx * x^y
        if tensor2.requires_grad:
            tensor2_grad = (grad * tensor1.tensor_data.log() * (tensor1.tensor_data ** tensor2.tensor_data))
            tensor2.backward(tensor2_grad)

    @staticmethod
    def scalar_pow(grad: TensorData, tensor1, value: T) -> None:
        if tensor1.requires_grad:
            tensor1_grad = (grad * (value * tensor1.tensor_data ** (value-1)))
            tensor1.backward(tensor1_grad)

    @staticmethod
    def exp(grad: TensorData, tensor1) -> None:
        if tensor1.requires_grad:
            tensor1_grad = (grad * tensor1.tensor_data.exp())
            tensor1.backward(tensor1_grad)

    @staticmethod
    def scalar_log(grad: TensorData, tensor1) -> None:
        if tensor1.requires_grad:
            tensor1.backward(grad / tensor1.tensor_data)

    @staticmethod
    def matmul(grad: TensorData, tensor1, tensor2) -> None:
        if tensor1.requires_grad:
            tensor1_grad = (grad @ tensor2.tensor_data.T)
            tensor1.backward(tensor1_grad)
        if tensor2.requires_grad:
            tensor2_grad = (tensor1.tensor_data.T @ grad)
            tensor2.backward(tensor2_grad)

    # potential bug since we don't factor in axes and keep dims
    @staticmethod
    def sum(grad: TensorData, tensor1) -> None:
        if tensor1.requires_grad:
            tensor1_grad = grad.broadcast(tensor1.tensor_data.shape())
            tensor1.backward(tensor1_grad)

    @staticmethod
    def swap(grad: TensorData, tensor1, axis1: int, axis2: int) -> None:
        if tensor1.requires_grad:
            tensor1.backward(grad.swap(axis1, axis2))

    @staticmethod
    def broadcast(grad: TensorData, tensor1) -> None:
        input_shape = tensor1.shape()
        new_shape = grad.shape()
        ndim_diff = len(new_shape) - len(input_shape)
        aligned_input_shape = [1] * ndim_diff + list(input_shape)

        sum_axes = []
        for i, (in_dim, out_dim) in enumerate(zip(aligned_input_shape, new_shape)):
            if in_dim == 1 and out_dim != 1:
                sum_axes.append(i)

        tensor1_grad = grad
        if sum_axes:
            tensor1_grad = grad.sum(sum_axes)

        tensor1_grad = tensor1_grad.reshape(input_shape)
        tensor1.backward(tensor1_grad)

    @staticmethod
    def reshape(grad: TensorData, tensor1) -> None:
        if tensor1.requires_grad:
            tensor1_grad = grad.reshape(tensor1.shape())
            tensor1.backward(tensor1_grad)

    @staticmethod
    def scalar_maximum(grad: TensorData, tensor1, val: T) -> None:
        if tensor1.requires_grad:
            mask = tensor1.tensor_data > val
            tensor1.backward(mask * grad)

    @staticmethod
    def scalar_minimum(grad: TensorData, tensor1, val: T) -> None:
        if tensor1.requires_grad:
            mask = tensor1.tensor_data < val
            tensor1.backward(mask * grad)

