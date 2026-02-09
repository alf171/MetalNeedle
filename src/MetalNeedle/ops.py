from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Any, TypeVar, Union, List

from MetalNeedle.data import TensorData
from MetalNeedle.grad import TensorGrad

if TYPE_CHECKING:
    from MetalNeedle.tensor import Tensor

LAZY_MODE = False
TENSOR_COUNTER = 0
T = TypeVar('T', bound=Union[int, float])

class TensorOperations:
    @staticmethod
    def add(tensor1: Tensor, tensor2: Tensor) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.add(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data + tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def scalar_add(tensor1: Tensor, value) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.scalar_add(grad, tensor1)
        tensor_data = tensor1.tensor_data + value
        return tensor_data, grad_fn

    @staticmethod
    def sub(tensor1: Tensor, tensor2: Tensor) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.sub(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data - tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def scalar_sub(tensor1: Tensor, value: T) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.scalar_sub(grad, tensor1)
        tensor_data = tensor1.tensor_data - value
        return tensor_data, grad_fn

    @staticmethod
    def mul(tensor1: Tensor, tensor2: Tensor) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.mul(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data * tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def scalar_mul(tensor1: Tensor, value: T) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.scalar_mul(grad, tensor1, value)
        tensor_data = tensor1.tensor_data * value
        return tensor_data, grad_fn

    @staticmethod
    def div(tensor1: Tensor, tensor2: Tensor) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.div(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data / tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def scalar_div(tensor1: Tensor, value: T) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.scalar_div(grad, tensor1, value)
        tensor_data = tensor1.tensor_data / value
        return tensor_data, grad_fn

    @staticmethod
    def pow(tensor1: Tensor, tensor2: Tensor) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.pow(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data ** tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def scalar_pow(tensor1: Tensor, value: T) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.scalar_pow(grad, tensor1, value)
        tensor_data = tensor1.tensor_data ** value
        return tensor_data, grad_fn

    @staticmethod
    def exp(tensor1: Tensor) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.exp(grad, tensor1)
        tensor_data = tensor1.tensor_data.exp()
        return tensor_data, grad_fn

    @staticmethod
    def scalar_log(tensor1: Tensor) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.scalar_log(grad, tensor1)
        tensor_data = tensor1.tensor_data.log()
        return tensor_data, grad_fn

    @staticmethod
    def matmul(tensor1: Tensor, tensor2: Tensor) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.matmul(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data @ tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def sum(tensor1: Tensor, axes, keep_dims) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.sum(grad, tensor1)
        tensor_data = tensor1.tensor_data.sum(axes, keep_dims)
        return tensor_data, grad_fn

    @staticmethod
    def swap(tensor1: Tensor, axis1: int, axis2: int) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.swap(grad, tensor1, axis1, axis2)
        tensor_data = tensor1.tensor_data.swap(axis1, axis2)
        return tensor_data, grad_fn

    @staticmethod
    def broadcast(tensor1: Tensor, new_shape: List[int]) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.broadcast(grad, tensor1)
        tensor_data = tensor1.tensor_data.broadcast(new_shape)
        return tensor_data, grad_fn

    @staticmethod
    def reshape(tensor1: Tensor, new_shape: List[int]) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.reshape(grad, tensor1)
        tensor_data = tensor1.tensor_data.reshape(new_shape)
        return tensor_data, grad_fn

    @staticmethod
    def scalar_maximum(tensor1: Tensor, val: T) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.scalar_maximum(grad, tensor1, val)
        tensor_data = tensor1.tensor_data.maximum(val)
        return tensor_data, grad_fn

    @staticmethod
    def scalar_minimum(tensor1: Tensor, val: T) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.scalar_minimum(grad, tensor1, val)
        tensor_data = tensor1.tensor_data.minimum(val)
        return tensor_data, grad_fn

    @staticmethod
    def maximum(tensor1: Tensor, tensor2: Tensor) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.maximum(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data.maximum(tensor2.tensor_data)
        return tensor_data, grad_fn

    @staticmethod
    def minimum(tensor1: Tensor, tensor2: Tensor) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.minimum(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data.minimum(tensor2.tensor_data)
        return tensor_data, grad_fn

    @staticmethod
    def max(tensor1: Tensor, axes: list[int], keep_dims: bool) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.max(grad, tensor)
        tensor_data = tensor.tensor_data.max(axes, keep_dims)
        return tensor_data, grad_fn

    @staticmethod
    def clip_scalar_scalar(tensor1: Tensor, lower_scalar: T, upper_scalar: T) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.clip_scalar_scalar(grad, tensor)
        tensor_data = tensor.tensor_data.clip(lower_scalar, upper_scalar)
        return tensor_data, grad_fn

    @staticmethod
    def clip_tensor_scalar(tensor1: Tensor, lower_tensor: Tensor, upper_scalar: T) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.clip_tensor_scalar(grad, tensor1, lower_tensor)
        tensor_data = tensor1.tensor_data.clip(lower_tensor.tensor_data, upper_scalar)
        return tensor_data, grad_fn

    @staticmethod
    def clip_scalar_tensor(tensor1: Tensor, lower_scalar: T, upper_tensor: Tensor) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.clip_tensor_scalar(grad, tensor1, upper_tensor)
        tensor_data = tensor1.tensor_data.clip(lower_scalar, upper_tensor.tensor_data)
        return tensor_data, grad_fn

    @staticmethod
    def clip_tensor_tensor(tensor1: Tensor, lower_tensor: Tensor, upper_tensor: Tensor) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.clip_tensor_tensor(grad, tensor1, lower_tensor, upper_tensor)
        tensor_data = tensor1.tensor_data.clip(lower_tensor.tensor_data, upper_tensor.tensor_data)
        return tensor_data, grad_fn

    @staticmethod
    def greater_than_tensor(tensor1: Tensor, tensor2: Tensor) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.greater_than_tensor(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data > tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def greater_than_scalar(tensor1: Tensor, val: T) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.greater_than_scalar(grad, tensor1, val)
        tensor_data = tensor1.tensor_data > val
        return tensor_data, grad_fn

    @staticmethod
    def greater_than_or_eq_tensor(tensor1: Tensor, tensor2: Tensor) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.greater_than_or_eq_tensor(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data >= tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def greater_than_or_eq_scalar(tensor1: Tensor, val: T) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.greater_than_or_eq_scalar(grad, tensor1, val)
        tensor_data = tensor1.tensor_data >= val
        return tensor_data, grad_fn

    @staticmethod
    def less_than_tensor(tensor1: Tensor, tensor2: Tensor) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.less_than_tensor(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data < tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def less_than_scalar(tensor1: Tensor, val: T) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.less_than_scalar(grad, tensor1, val)
        tensor_data = tensor1.tensor_data < val
        return tensor_data, grad_fn

    @staticmethod
    def less_than_or_eq_tensor(tensor1: Tensor, tensor2: Tensor) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.less_than_or_eq_tensor(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data <= tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def less_than_or_eq_scalar(tensor1: Tensor, val: T) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.less_than_or_eq_scalar(grad, tensor1, val)
        tensor_data = tensor1.tensor_data <= val
        return tensor_data, grad_fn

    @staticmethod
    def eq_tensor(tensor1: Tensor, tensor2: Tensor) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.eq_tensor(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data == tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def eq_scalar(tensor1: Tensor, val: T) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.eq_scalar(grad, tensor1, val)
        tensor_data = tensor1.tensor_data == val
        return tensor_data, grad_fn

    @staticmethod
    def neq_tensor(tensor1: Tensor, tensor2: Tensor) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.neq_tensor(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data != tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def neq_scalar(tensor1: Tensor, val: T) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.neq_scalar(grad, tensor1, val)
        tensor_data = tensor1.tensor_data != val
        return tensor_data, grad_fn
