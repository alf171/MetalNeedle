from typing import Tuple, Any, TypeVar, Union, List

from MetalNeedle.data import TensorData
from MetalNeedle.grad import TensorGrad

LAZY_MODE = False
TENSOR_COUNTER = 0
T = TypeVar('T', bound=Union[int, float])

# TODO: might be a decent refactor but only taking in TensorData
# seems like a better abstraction
# TODO: move all grad fns into their own file
class TensorOperations:
    @staticmethod
    def add(tensor1, tensor2) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.add(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data + tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def scalar_add(tensor1, value) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.scalar_add(grad, tensor1)
        tensor_data = tensor1.tensor_data + value
        return tensor_data, grad_fn

    @staticmethod
    def sub(tensor1, tensor2) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.sub(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data - tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def scalar_sub(tensor1, value) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.scalar_sub(grad, tensor1)
        tensor_data = tensor1.tensor_data - value
        return tensor_data, grad_fn

    @staticmethod
    def mul(tensor1, tensor2) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.mul(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data * tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def scalar_mul(tensor1, value: T) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.scalar_mul(grad, tensor1, value)
        tensor_data = tensor1.tensor_data * value
        return tensor_data, grad_fn

    @staticmethod
    def div(tensor1, tensor2) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.div(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data / tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def scalar_div(tensor1, value: T) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.scalar_div(grad, tensor1, value)
        tensor_data = tensor1.tensor_data / value
        return tensor_data, grad_fn

    @staticmethod
    def pow(tensor1, tensor2) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.pow(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data ** tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def scalar_pow(tensor1, value) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.scalar_pow(grad, tensor1, value)
        tensor_data = tensor1.tensor_data ** value
        return tensor_data, grad_fn

    @staticmethod
    def exp(tensor1) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.exp(grad, tensor1)
        tensor_data = tensor1.tensor_data.exp()
        return tensor_data, grad_fn

    @staticmethod
    def scalar_log(tensor1) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.scalar_log(grad, tensor1)
        tensor_data = tensor1.tensor_data.log()
        return tensor_data, grad_fn

    @staticmethod
    def matmul(tensor1, tensor2) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.matmul(grad, tensor1, tensor2)
        tensor_data = tensor1.tensor_data @ tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def sum(tensor1, axes, keep_dims) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.sum(grad, tensor1)
        tensor_data = tensor1.tensor_data.sum(axes, keep_dims)
        return tensor_data, grad_fn

    @staticmethod
    def swap(tensor1, axis1: int, axis2: int) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.swap(tensor1. axis1, axis2)
        tensor_data = tensor1.tensor_data.swap(axis1, axis2)
        return tensor_data, grad_fn

    @staticmethod
    def broadcast(tensor1, new_shape: List[int]) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.broadcast(grad, tensor1)
        tensor_data = tensor1.tensor_data.broadcast(new_shape)
        return tensor_data, grad_fn

    @staticmethod
    def reshape(tensor1, new_shape: List[int]) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.reshape(grad, tensor1)
        tensor_data = tensor1.tensor_data.reshape(new_shape)
        return tensor_data, grad_fn

    @staticmethod
    def scalar_maximum(tensor1, val: T) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.scalar_maximum(grad, tensor1, val)
        tensor_data = tensor1.tensor_data.maximum(val)
        return tensor_data, grad_fn

    @staticmethod
    def scalar_minimum(tensor1, val: T) -> Tuple[TensorData, Any]:
        grad_fn = lambda grad: TensorGrad.scalar_minimum(grad, tensor1, val)
        tensor_data = tensor1.tensor_data.minimum(val)
        return tensor_data, grad_fn

    @staticmethod
    def maximum(tensor1, tensor2) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                raise NotImplemented("maximum back not implemented")
            if tensor2.requires_grad:
                raise NotImplemented("maximum back not implemented")
        tensor_data = tensor1.tensor_data.maximum(tensor2.tensor_data)
        return tensor_data, _grad_fn

    @staticmethod
    def minimum(tensor1, tensor2) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                raise NotImplemented("minimum back not implemented")
            if tensor2.requires_grad:
                raise NotImplemented("minimum back not implemented")
        tensor_data = tensor1.tensor_data.minimum(tensor2.tensor_data)
        return tensor_data, _grad_fn

    @staticmethod
    def max(tensor, axes, keep_dims) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor.requires_grad:
                # During forward, save the argmax locations (where the max happened).
                # During backward, create a mask (1s where input == max, 0s elsewhere).
                # Multiply incoming grad by the mask.
                # Then broadcast it back to the original input shape if needed.
                raise NotImplemented("max back not implemented")

        tensor_data = tensor.tensor_data.max(axes, keep_dims)
        return tensor_data, _grad_fn

    @staticmethod
    def clip_scalar_scalar(tensor, lower_scalar, upper_scalar) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor.requires_grad:
                raise NotImplemented("clip back not implemented")

        tensor_data = tensor.tensor_data.clip(lower_scalar, upper_scalar)
        return tensor_data, _grad_fn

    @staticmethod
    def clip_tensor_scalar(tensor, lower_tensor, upper_scalar) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor.requires_grad:
                raise NotImplemented("clip back not implemented")
            if lower_tensor.requires_grad:
                raise NotImplemented("clip back not implemented")

        tensor_data = tensor.tensor_data.clip(lower_tensor.tensor_data, upper_scalar)
        return tensor_data, _grad_fn

    @staticmethod
    def clip_scalar_tensor(tensor, lower_scalar, upper_tensor) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor.requires_grad:
                raise NotImplemented("clip back not implemented")
            if upper_tensor.requires_grad:
                raise NotImplemented("clip back not implemented")

        tensor_data = tensor.tensor_data.clip(lower_scalar, upper_tensor.tensor_data)
        return tensor_data, _grad_fn

    @staticmethod
    def clip_tensor_tensor(tensor, lower_tensor, upper_tensor) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor.requires_grad:
                raise NotImplemented("clip back not implemented")
            if lower_tensor.requires_grad:
                raise NotImplemented("clip back not implemented")
            if upper_tensor.requires_grad:
                raise NotImplemented("clip back not implemented")

        tensor_data = tensor.tensor_data.clip(lower_tensor.tensor_data, upper_tensor.tensor_data)
        return tensor_data, _grad_fn

    @staticmethod
    def greater_than_tensor(tensor1, tensor2) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                mask = tensor1.tensor_data > tensor2.tensor_data
                tensor1.backward(mask * grad)

        tensor_data = tensor1.tensor_data > tensor2.tensor_data
        return tensor_data, _grad_fn

    @staticmethod
    def greater_than_scalar(tensor, val) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor.requires_grad:
                mask = tensor.tensor_data > val
                tensor.backward(mask * grad)

        tensor_data = tensor.tensor_data > val
        return tensor_data, _grad_fn

    @staticmethod
    def greater_than_or_eq_tensor(tensor1, tensor2) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1_grad = grad >= tensor2.tensor_data
                tensor1.backward(tensor1_grad)
            if tensor2.requires_grad:
                tensor2_grad = grad >= tensor1.tensor_data
                tensor2.backward(tensor2_grad)

        tensor_data = tensor1.tensor_data >= tensor2.tensor_data
        return tensor_data, _grad_fn

    @staticmethod
    def greater_than_or_eq_scalar(tensor, val) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor.requires_grad:
                tensor_grad = grad >= val
                tensor.backward(tensor_grad)

        tensor_data = tensor.tensor_data >= val
        return tensor_data, _grad_fn

    @staticmethod
    def less_than_tensor(tensor1, tensor2) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1_grad = grad < tensor2.tensor_data
                tensor1.backward(tensor1_grad)
            if tensor2.requires_grad:
                tensor2_grad = grad < tensor1.tensor_data
                tensor2.backward(tensor2_grad)

        tensor_data = tensor1.tensor_data < tensor2.tensor_data
        return tensor_data, _grad_fn

    @staticmethod
    def less_than_scalar(tensor, val) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor.requires_grad:
                tensor_grad = grad < val
                tensor.backward(tensor_grad)

        tensor_data = tensor.tensor_data < val
        return tensor_data, _grad_fn

    @staticmethod
    def less_than_or_eq_tensor(tensor1, tensor2) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1_grad = grad <= tensor2.tensor_data
                tensor1.backward(tensor1_grad)
            if tensor2.requires_grad:
                tensor2_grad = grad <= tensor1.tensor_data
                tensor2.backward(tensor2_grad)

        tensor_data = tensor1.tensor_data <= tensor2.tensor_data
        return tensor_data, _grad_fn

    @staticmethod
    def less_than_or_eq_scalar(tensor, val) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor.requires_grad:
                tensor_grad = grad <= val
                tensor.backward(tensor_grad)

        tensor_data = tensor.tensor_data <= val
        return tensor_data, _grad_fn

    @staticmethod
    def eq_tensor(tensor1, tensor2) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1_grad = grad == tensor2.tensor_data
                tensor1.backward(tensor1_grad)
            if tensor2.requires_grad:
                tensor2_grad = grad == tensor1.tensor_data
                tensor2.backward(tensor2_grad)

        tensor_data = tensor1.tensor_data == tensor2.tensor_data
        return tensor_data, _grad_fn

    @staticmethod
    def eq_scalar(tensor, val) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor.requires_grad:
                tensor_grad = grad.eq(val)
                tensor.backward(tensor_grad)

        tensor_data = tensor.tensor_data == val
        return tensor_data, _grad_fn

    @staticmethod
    def neq_tensor(tensor1, tensor2) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1_grad = grad != tensor2.tensor_data
                tensor1.backward(tensor1_grad)
            if tensor2.requires_grad:
                tensor2_grad = grad != tensor1.tensor_data
                tensor2.backward(tensor2_grad)

        tensor_data = tensor1.tensor_data != tensor2.tensor_data
        return tensor_data, _grad_fn

    @staticmethod
    def neq_scalar(tensor, val) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor.requires_grad:
                tensor_grad = grad != val
                tensor.backward(tensor_grad)

        tensor_data = tensor.tensor_data != val
        return tensor_data, _grad_fn

    @staticmethod
    def _log_before_grad(op, grad, tensor1=None, tensor2=None) -> None:
        # Log information about the incoming gradient
        print(f"[{op} BACKWARD] Gradient shape: {grad.shape() if hasattr(grad, 'shape') else 'scalar'}")
        print(f"[{op} BACKWARD] Gradient value: {grad.data() if hasattr(grad, 'data') else grad}")

        # Log information about the tensors being added
        if tensor1 is not None:
            t1_name = tensor1._debug_name() or "unnamed_tensor1"
            print(f"[{op} BACKWARD] Tensor1 '{t1_name}' shape: {tensor1.shape()}, requires_grad: {tensor1.requires_grad}")
        if tensor2 is not None:
            t2_name = tensor2._debug_name() or "unnamed_tensor2"
            print(f"[{op} BACKWARD] Tensor2 '{t2_name}' shape: {tensor2.shape()}, requires_grad: {tensor2.requires_grad}")

        # Log the current gradient values of both tensors before update
        if tensor1 is not None:
            print(f"[{op} BACKWARD] Tensor1 '{t1_name}' grad before: {tensor1.grad.data() if tensor1.grad is not None else None}")
        if tensor2 is not None:
            print(f"[{op} BACKWARD] Tensor2 '{t2_name}' grad before: {tensor2.grad.data() if tensor2.grad is not None else None}")

    @staticmethod
    def _log_after_grad(op, tensor1, tensor2) -> None:
        t1_name = tensor1._debug_name() or "unnamed_tensor1"
        t2_name = tensor2._debug_name() or "unnamed_tensor2"
        print(f"[{op} BACKWARD] Tensor1 '{t1_name}' grad after: {tensor1.grad.data() if tensor1.grad is not None else None}")
        print(f"[{op} BACKWARD] Tensor2 '{t2_name}' grad after: {tensor2.grad.data() if tensor2.grad is not None else None}")