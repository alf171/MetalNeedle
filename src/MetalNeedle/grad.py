from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Union

from MetalNeedle.data import TensorData

if TYPE_CHECKING:
    from MetalNeedle.tensor import Tensor

T = TypeVar("T", bound=Union[int, float])

class TensorGrad:
    @staticmethod
    def add(grad: TensorData, tensor1: Tensor, tensor2: Tensor) -> None:
        if tensor1.requires_grad:
            tensor1.backward(grad)
        if tensor2.requires_grad:
            tensor2.backward(grad)

    @staticmethod
    def scalar_add(grad: TensorData, tensor1: Tensor) -> None:
        if tensor1.requires_grad:
            tensor1.backward(grad)

    @staticmethod
    def sub(grad: TensorData, tensor1: Tensor, tensor2: Tensor) -> None:
        if tensor1.requires_grad:
            tensor1.backward(grad)
        if tensor2.requires_grad:
            tensor2_grad = -grad
            tensor2.backward(tensor2_grad)

    @staticmethod
    def scalar_sub(grad: TensorData, tensor1: Tensor) -> None:
        if tensor1.requires_grad:
            tensor1.backward(grad)

    @staticmethod
    def mul(grad: TensorData, tensor1: Tensor, tensor2: Tensor) -> None:
        if tensor1.requires_grad:
            tensor1_grad = tensor2.tensor_data * grad
            tensor1.backward(tensor1_grad)
        if tensor2.requires_grad:
            tensor2_grad = tensor1.tensor_data * grad
            tensor2.backward(tensor2_grad)

    @staticmethod
    def scalar_mul(grad: TensorData, tensor1: Tensor, value: T) -> None:
        if tensor1.requires_grad:
            tensor1.backward(grad * value)

    @staticmethod
    def div(grad: TensorData, tensor1: Tensor, tensor2: Tensor) -> None:
        # da(A/B) = 1/B
        if tensor1.requires_grad:
            tensor1_grad = grad / tensor2.tensor_data
            tensor1.backward(tensor1_grad)
        # db(A/B) = -A/B^2
        if tensor2.requires_grad:
            tensor2_grad = (-grad * tensor1.tensor_data) / (tensor2.tensor_data**2)
            tensor2.backward(tensor2_grad)

    @staticmethod
    def scalar_div(grad: TensorData, tensor1: Tensor, value: T) -> None:
        if tensor1.requires_grad:
            tensor1.backward(grad / value)

    @staticmethod
    def pow(grad: TensorData, tensor1: Tensor, tensor2: Tensor) -> None:
        # dx(x^y) = y * x^(y-1)
        if tensor1.requires_grad:
            tensor1_grad = (
                grad
                * tensor2.tensor_data
                * (tensor1.tensor_data ** (tensor2.tensor_data - 1))
            )
            tensor1.backward(tensor1_grad)
        # dy(x^y) = dy(e^(y*lnx)) = lnx*e^(y*lnx) = lnx * x^y
        if tensor2.requires_grad:
            tensor2_grad = (
                grad
                * tensor1.tensor_data.log()
                * (tensor1.tensor_data**tensor2.tensor_data)
            )
            tensor2.backward(tensor2_grad)

    @staticmethod
    def scalar_pow(grad: TensorData, tensor1: Tensor, value: T) -> None:
        if tensor1.requires_grad:
            tensor1_grad = grad * (value * tensor1.tensor_data) ** (value - 1)
            tensor1.backward(tensor1_grad)

    @staticmethod
    def exp(grad: TensorData, tensor1: Tensor) -> None:
        if tensor1.requires_grad:
            tensor1_grad = grad * tensor1.tensor_data.exp()
            tensor1.backward(tensor1_grad)

    @staticmethod
    def scalar_log(grad: TensorData, tensor1: Tensor) -> None:
        if tensor1.requires_grad:
            tensor1.backward(grad / tensor1.tensor_data)

    @staticmethod
    def matmul(grad: TensorData, tensor1: Tensor, tensor2: Tensor) -> None:
        if tensor1.requires_grad:
            tensor1_grad = grad @ tensor2.tensor_data.T
            tensor1.backward(tensor1_grad)
        if tensor2.requires_grad:
            tensor2_grad = tensor1.tensor_data.T @ grad
            tensor2.backward(tensor2_grad)

    @staticmethod
    def sum(grad: TensorData, tensor1: Tensor, axes: list[int], keep_dims: bool) -> None:
        if tensor1.requires_grad:
            if not keep_dims:
                expanded_shape = tensor1.tensor_data.shape()[:]
                for axis in axes:
                    expanded_shape[axis] = 1
                grad = grad.reshape(expanded_shape)
            tensor1_grad = grad.broadcast(tensor1.tensor_data.shape())
            tensor1.backward(tensor1_grad)

    @staticmethod
    def swap(grad: TensorData, tensor1: Tensor, axis1: int, axis2: int) -> None:
        if tensor1.requires_grad:
            tensor1.backward(grad.swap(axis1, axis2))

    @staticmethod
    def broadcast(grad: TensorData, tensor1: Tensor) -> None:
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
    def reshape(grad: TensorData, tensor1: Tensor) -> None:
        if tensor1.requires_grad:
            tensor1_grad = grad.reshape(tensor1.shape())
            tensor1.backward(tensor1_grad)

    @staticmethod
    def scalar_maximum(grad: TensorData, tensor1: Tensor, val: T) -> None:
        if tensor1.requires_grad:
            mask = tensor1.tensor_data > val
            tensor1.backward(mask * grad)

    @staticmethod
    def scalar_minimum(grad: TensorData, tensor1: Tensor, val: T) -> None:
        if tensor1.requires_grad:
            mask = tensor1.tensor_data < val
            tensor1.backward(mask * grad)

    # TODO: impl with where(cond, a, b) to reduce boiler plate
    @staticmethod
    def maximum(grad: TensorData, tensor1: Tensor, tensor2) -> None:
        # dx(max(x,y)) = {grad if x > y, else 0 }
        if tensor1.requires_grad:
            mask1 = tensor1.tensor_data > tensor2.tensor_data
            mask_equal = tensor1.tensor_data == tensor2.tensor_data
            grad_half = grad * 0.5
            tensor1.backward(mask1 * grad + mask_equal * grad_half)
        # dy(max(x,y)) = {grad if x < y, else 0 }
        if tensor2.requires_grad:
            mask2 = tensor1.tensor_data < tensor2.tensor_data
            mask_equal = tensor1.tensor_data == tensor2.tensor_data
            grad_half = grad * 0.5
            tensor2.backward(mask2 * grad + mask_equal * grad_half)

    @staticmethod
    def minimum(grad: TensorData, tensor1: Tensor, tensor2: Tensor) -> None:
        # dx(min(x,y)) = {grad if x < y, else 0 }
        if tensor1.requires_grad:
            mask1 = tensor1.tensor_data < tensor2.tensor_data
            mask_equal = tensor1.tensor_data == tensor2.tensor_data
            grad_half = grad * 0.5
            tensor1.backward(mask1 * grad + mask_equal * grad_half)
        # dy(min(x,y)) = {grad if x > y, else 0 }
        if tensor2.requires_grad:
            mask2 = tensor1.tensor_data > tensor2.tensor_data
            mask_equal = tensor1.tensor_data == tensor2.tensor_data
            grad_half = grad * 0.5
            tensor2.backward(mask2 * grad + mask_equal * grad_half)

    @staticmethod
    def max(grad: TensorData, tensor1: Tensor, axes: list[int], keep_dims: bool) -> None:
        if not tensor1.requires_grad:
            return
        # dx(max(x)) = (1[x = max(x)] / num_max_ties) * grad
        max_vals = tensor1.tensor_data.max(axes, keep_dims)
        if not keep_dims:
            expanded_shape = tensor1.tensor_data.shape()[:]
            for axis in axes:
                expanded_shape[axis] = 1
            grad = grad.reshape(expanded_shape)
            max_vals = max_vals.reshape(expanded_shape)

        grad = grad.broadcast(tensor1.tensor_data.shape())
        max_vals = max_vals.broadcast(tensor1.tensor_data.shape())

        mask = tensor1.tensor_data == max_vals
        tensor1.backward(mask * grad)

    @staticmethod
    def clip_scalar_scalar(grad: TensorData, tensor1: Tensor, lower_scalar: T, upper_scalar: T) -> None:
        if tensor1.requires_grad:
            mask = (tensor1.tensor_data >= lower_scalar) and  (tensor1.tensor_data <= upper_scalar)
            tensor1.backward(mask * grad)

    @staticmethod
    def clip_tensor_scalar(grad: TensorData, tensor: Tensor, lower_tensor: Tensor) -> None:
        if tensor.requires_grad:
            raise NotImplementedError("clip back not implemented")
        if lower_tensor.requires_grad:
            raise NotImplementedError("clip back not implemented")

    @staticmethod
    def clip_scalar_tensor(grad: TensorData, tensor: Tensor, upper_tensor: Tensor) -> None:
        if tensor.requires_grad:
            raise NotImplementedError("clip back not implemented")
        if upper_tensor.requires_grad:
            raise NotImplementedError("clip back not implemented")

    @staticmethod
    def clip_tensor_tensor(
        grad: TensorData, tensor1: Tensor, lower_tensor, upper_tensor
    ) -> None:
        if tensor1.requires_grad:
            raise NotImplementedError("clip back not implemented")
        if lower_tensor.requires_grad:
            raise NotImplementedError("clip back not implemented")
        if upper_tensor.requires_grad:
            raise NotImplementedError("clip back not implemented")

    @staticmethod
    def greater_than_tensor(grad: TensorData, tensor1: Tensor, tensor2: Tensor) -> None:
        if tensor1.requires_grad:
            mask = tensor1.tensor_data > tensor2.tensor_data
            tensor1.backward(mask * grad)

    @staticmethod
    def greater_than_scalar(grad: TensorData, tensor1: Tensor, val: T) -> None:
        if tensor1.requires_grad:
            mask = tensor1.tensor_data > val
            tensor1.backward(mask * grad)

    @staticmethod
    def greater_than_or_eq_tensor(grad: TensorData, tensor1: Tensor, tensor2: Tensor) -> None:
        if tensor1.requires_grad:
            tensor1_grad = grad >= tensor2.tensor_data
            tensor1.backward(tensor1_grad)
        if tensor2.requires_grad:
            tensor2_grad = grad >= tensor1.tensor_data
            tensor2.backward(tensor2_grad)

    @staticmethod
    def greater_than_or_eq_scalar(grad: TensorData, tensor1: Tensor, val: T) -> None:
        if tensor1.requires_grad:
            tensor_grad = grad >= val
            tensor1.backward(tensor_grad)

    @staticmethod
    def less_than_tensor(grad: TensorData, tensor1: Tensor, tensor2: Tensor) -> None:
        if tensor1.requires_grad:
            tensor1_grad = grad < tensor2.tensor_data
            tensor1.backward(tensor1_grad)
        if tensor2.requires_grad:
            tensor2_grad = grad < tensor1.tensor_data
            tensor2.backward(tensor2_grad)

    @staticmethod
    def less_than_scalar(grad: TensorData, tensor1: Tensor, val) -> None:
        if tensor1.requires_grad:
            tensor_grad = grad < val
            tensor1.backward(tensor_grad)

    @staticmethod
    def less_than_or_eq_tensor(grad: TensorData, tensor1: Tensor, tensor2: Tensor) -> None:
        if tensor1.requires_grad:
            tensor1_grad = grad <= tensor2.tensor_data
            tensor1.backward(tensor1_grad)
        if tensor2.requires_grad:
            tensor2_grad = grad <= tensor1.tensor_data
            tensor2.backward(tensor2_grad)

    @staticmethod
    def less_than_or_eq_scalar(grad: TensorData, tensor1: Tensor, val: T) -> None:
        if tensor1.requires_grad:
            tensor_grad = grad <= val
            tensor1.backward(tensor_grad)

    @staticmethod
    def eq_tensor(grad: TensorData, tensor1: Tensor, tensor2: Tensor) -> None:
        if tensor1.requires_grad:
            tensor1_grad = grad == tensor2.tensor_data
            tensor1.backward(tensor1_grad)
        if tensor2.requires_grad:
            tensor2_grad = grad == tensor1.tensor_data
            tensor2.backward(tensor2_grad)

    @staticmethod
    def eq_scalar(grad: TensorData, tensor1: Tensor, val: T) -> None:
        if tensor1.requires_grad:
            tensor_grad = grad == val
            tensor1.backward(tensor_grad)

    @staticmethod
    def neq_tensor(grad: TensorData, tensor1: Tensor, tensor2: Tensor) -> None:
        if tensor1.requires_grad:
            tensor1_grad = grad != tensor2.tensor_data
            tensor1.backward(tensor1_grad)
        if tensor2.requires_grad:
            tensor2_grad = grad != tensor1.tensor_data
            tensor2.backward(tensor2_grad)

    @staticmethod
    def neq_scalar(grad: TensorData, tensor1: Tensor, val: T) -> None:
        if tensor1.requires_grad:
            tensor_grad = grad != val
            tensor1.backward(tensor_grad)

    @staticmethod
    def _log_before_grad(op: str, grad, tensor1: Tensor | None = None, tensor2: Tensor | None = None) -> None:
        # Log information about the incoming gradient
        print(
            f"[{op} BACKWARD] Gradient shape: {grad.shape() if hasattr(grad, 'shape') else 'scalar'}"
        )
        print(
            f"[{op} BACKWARD] Gradient value: {grad.data() if hasattr(grad, 'data') else grad}"
        )

        # Log information about the tensors being added
        if tensor1 is not None:
            t1_name = tensor1.debug_name() or "unnamed_tensor1"
            print(
                f"[{op} BACKWARD] Tensor1 '{t1_name}' shape: {tensor1.shape()}, requires_grad: {tensor1.requires_grad}"
            )
        if tensor2 is not None:
            t2_name = tensor2.debug_name() or "unnamed_tensor2"
            print(
                f"[{op} BACKWARD] Tensor2 '{t2_name}' shape: {tensor2.shape()}, requires_grad: {tensor2.requires_grad}"
            )

        # Log the current gradient values of both tensors before update
        if tensor1 is not None:
            t1_name = tensor1.debug_name() or "unnamed_tensor1"
            print(
                f"[{op} BACKWARD] Tensor1 '{t1_name}' grad before: {tensor1.grad.data() if tensor1.grad is not None else None}"
            )
        if tensor2 is not None:
            t2_name = tensor2.debug_name() or "unnamed_tensor2"
            print(
                f"[{op} BACKWARD] Tensor2 '{t2_name}' grad before: {tensor2.grad.data() if tensor2.grad is not None else None}"
            )

    @staticmethod
    def _log_after_grad(op: str, tensor1: Tensor, tensor2: Tensor) -> None:
        t1_name = tensor1.debug_name() or "unnamed_tensor1"
        t2_name = tensor2.debug_name() or "unnamed_tensor2"
        print(
            f"[{op} BACKWARD] Tensor1 '{t1_name}' grad after: {tensor1.grad.data() if tensor1.grad is not None else None}"
        )
        print(
            f"[{op} BACKWARD] Tensor2 '{t2_name}' grad after: {tensor2.grad.data() if tensor2.grad is not None else None}"
        )
