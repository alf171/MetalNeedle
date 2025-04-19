from typing import Tuple, Any

from MetalNeedle.data import TensorData

LAZY_MODE = False
TENSOR_COUNTER = 0

# TODO: might be a decent refactor but only taking in TensorData
# seems like a better abstraction
class TensorOperations:
    @staticmethod
    def add(tensor1, tensor2) -> Tuple[TensorData, Any]:
        def grad_fn(grad):
            if tensor1.requires_grad:
                tensor1.backward(grad)
            if tensor2.requires_grad:
                tensor2.backward(grad)

        tensor_data = tensor1.tensor_data + tensor2.tensor_data
        return tensor_data, grad_fn

    @staticmethod
    def scalar_add(tensor1, value) -> Tuple[TensorData, Any]:
        def grad_fn(grad):
            if tensor1.requires_grad:
                tensor1.backward(grad)

        tensor_data = tensor1.tensor_data + value
        return tensor_data, grad_fn

    @staticmethod
    def sub(tensor1, tensor2) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1.backward(grad)
            if tensor2.requires_grad:
                tensor2_grad = (-1 * grad)
                tensor2.backward(tensor2_grad)

        tensor_data = tensor1.tensor_data - tensor2.tensor_data
        return tensor_data, _grad_fn

    @staticmethod
    def scalar_sub(tensor1, value) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1.backward(grad)
        tensor_data = tensor1.tensor_data - value
        return tensor_data, _grad_fn

    @staticmethod
    def mul(tensor1, tensor2) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1_grad = (tensor2.tensor_data * grad)
                tensor1.backward(tensor1_grad)
            if tensor2.requires_grad:
                tensor2_grad = (tensor1.tensor_data * grad)
                tensor2.backward(tensor2_grad)

        tensor_data = tensor1.tensor_data * tensor2.tensor_data
        return tensor_data, _grad_fn

    @staticmethod
    def scalar_mul(tensor1, value) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1.grad = (grad * value)

        tensor_data = tensor1.tensor_data * value
        return tensor_data, _grad_fn

    @staticmethod
    def div(tensor1, tensor2) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            # da(A/B) = 1/B
            if tensor1.requires_grad:
                tensor1_grad = (grad / tensor2.tensor_data)
                tensor1.backward(tensor1_grad)
            # db(A/B) = -A/B^2
            if tensor2.requires_grad:
                tensor2_grad = (-grad * tensor1.tensor_data) / (tensor2.tensor_data ** 2)
                tensor2.backward(tensor2_grad)

        tensor_data = tensor1.tensor_data / tensor2.tensor_data
        return tensor_data, _grad_fn

    @staticmethod
    def scalar_div(tensor1, value) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1.backward(grad / value)
        tensor_data = tensor1.tensor_data / value
        return tensor_data, _grad_fn

    @staticmethod
    def pow(tensor1, tensor2) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            # dx(x^y) = y * x^(y-1)
            if tensor1.requires_grad:
                tensor1_grad = (grad * tensor2.tensor_data * (tensor1.tensor_data ** tensor2.tensor_data))
                tensor1.backward(tensor1_grad)
            # dy(x^y) = dy(e^(y*lnx)) = lnx*e^(y*lnx) = lnx * x^y
            if tensor2.requires_grad:
                tensor2_grad = (grad * tensor1.tensor_data.log() * (tensor1.tensor_data ** tensor2.tensor_data))
                tensor2.backward(tensor2_grad)

        tensor_data = tensor1.tensor_data ** tensor2.tensor_data
        return tensor_data, _grad_fn


    @staticmethod
    def scalar_pow(tensor1, value) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1_grad = (grad * (value * tensor1.tensor_data ** (value-1)))
                tensor1.grad(tensor1_grad)

        tensor_data = tensor1.tensor_data ** value
        return tensor_data, _grad_fn

    @staticmethod
    def exp(tensor1) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1_grad = (grad * tensor1.tensor_data.exp())
                tensor1.grad(tensor1_grad)

        tensor_data = tensor1.tensor_data.exp()
        return tensor_data, _grad_fn

    @staticmethod
    def scalar_log(tensor1) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1.backward(grad / tensor1.tensor_data)

        tensor_data = tensor1.tensor_data.log()
        return tensor_data, _grad_fn

    @staticmethod
    def matmul(tensor1, tensor2) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1_grad = (grad @ tensor2.tensor_data.T)
                tensor1.backward (tensor1_grad)
            if tensor2.requires_grad:
                tensor2_grad = (tensor1.tensor_data.T @ grad)
                tensor2.backward(tensor2_grad)

        tensor_data = tensor1.tensor_data @ tensor2.tensor_data
        return tensor_data, _grad_fn

    @staticmethod
    def sum(tensor1, axes, keep_dims) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1_grad = grad.broadcast(tensor1.tensor_data.shape())
                tensor1.backward(tensor1_grad)

        tensor_data = tensor1.tensor_data.sum(axes, keep_dims)
        return tensor_data, _grad_fn

    @staticmethod
    def swap(tensor1, axis1, axis2) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1.grad(grad.swap(axis1, axis2))

        tensor_data = tensor1.tensor_data.swap(axis1, axis2)
        return tensor_data, _grad_fn

    @staticmethod
    def broadcast(tensor1, new_shape) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                sum_dims = []
                for i, (ts, ns) in enumerate(zip(tensor1.shape(), new_shape)):
                    if ts != ns:
                        sum_dims.append(i)
                tensor1_grad = grad.ones_like().sum(sum_dims)
                tensor1.backward(tensor1_grad)

        tensor_data = tensor1.tensor_data.broadcast(new_shape)
        return tensor_data, _grad_fn

    @staticmethod
    def reshape(tensor1, new_shape) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                tensor1_grad = grad.reshape(tensor1.shape())
                tensor1.backward(tensor1_grad)

        tensor_data = tensor1.tensor_data.reshape(new_shape)
        return tensor_data, _grad_fn

    # TODO: need support for operators like >= on tensors
    # in order to implement backwards
    @staticmethod
    def scalar_maximum(tensor, val) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor.requires_grad:
                pass
        tensor_data = tensor.tensor_data.maximum(val)
        return tensor_data, _grad_fn

    # TODO: need support for operators like >= on tensors
    # in order to implement backwards
    @staticmethod
    def scalar_minimum(tensor, val) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor.requires_grad:
                pass
        tensor_data = tensor.tensor_data.minimum(val)
        return tensor_data, _grad_fn

    # TODO: need support for operators like >= on tensors
    # in order to implement backwards
    @staticmethod
    def maximum(tensor1, tensor2) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                pass
            if tensor2.requires_grad:
                pass
        tensor_data = tensor1.tensor_data.maximum(tensor2.tensor_data)
        return tensor_data, _grad_fn

    @staticmethod
    def minimum(tensor1, tensor2) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor1.requires_grad:
                pass
            if tensor2.requires_grad:
                pass
        tensor_data = tensor1.tensor_data.minimum(tensor2.tensor_data)
        return tensor_data, _grad_fn

    @staticmethod
    def max(tensor, axes, keep_dims) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor.requires_grad:
                pass

        tensor_data = tensor.tensor_data.max(axes, keep_dims)
        return tensor_data, _grad_fn

    @staticmethod
    def clip_scalar_scalar(tensor, lower_scalar, upper_scalar) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor.requires_grad:
                pass

        tensor_data = tensor.tensor_data.clip(lower_scalar, upper_scalar)
        return tensor_data, _grad_fn

    @staticmethod
    def clip_tensor_scalar(tensor, lower_tensor, upper_scalar) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor.requires_grad:
                pass
            if lower_tensor.requires_grad:
                pass

        tensor_data = tensor.tensor_data.clip(lower_tensor.tensor_data, upper_scalar)
        return tensor_data, _grad_fn

    @staticmethod
    def clip_scalar_tensor(tensor, lower_scalar, upper_tensor) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor.requires_grad:
                pass
            if upper_tensor.requires_grad:
                pass

        tensor_data = tensor.tensor_data.clip(lower_scalar, upper_tensor.tensor_data)
        return tensor_data, _grad_fn

    @staticmethod
    def clip_tensor_tensor(tensor, lower_tensor, upper_tensor) -> Tuple[TensorData, Any]:
        def _grad_fn(grad):
            if tensor.requires_grad:
                pass
            if lower_tensor.requires_grad:
                pass
            if upper_tensor.requires_grad:
                pass

        tensor_data = tensor.tensor_data.clip(lower_tensor.tensor_data, upper_tensor.tensor_data)
        return tensor_data, _grad_fn


    @staticmethod
    def _log_before_grad(op, grad, tensor1, tensor2) -> None:
        # Log information about the incoming gradient
        print(f"[{op} BACKWARD] Gradient shape: {grad.shape() if hasattr(grad, 'shape') else 'scalar'}")
        print(f"[{op} BACKWARD] Gradient value: {grad.data() if hasattr(grad, 'data') else grad}")

        # Log information about the tensors being added
        t1_name = tensor1._debug_name() or "unnamed_tensor1"
        t2_name = tensor2._debug_name() or "unnamed_tensor2"
        print(f"[{op} BACKWARD] Tensor1 '{t1_name}' shape: {tensor1.shape()}, requires_grad: {tensor1.requires_grad}")
        print(f"[{op} BACKWARD] Tensor2 '{t2_name}' shape: {tensor2.shape()}, requires_grad: {tensor2.requires_grad}")

        # Log the current gradient values of both tensors before update
        print(f"[{op} BACKWARD] Tensor1 '{t1_name}' grad before: {tensor1.grad.data() if tensor1.grad is not None else None}")
        print(f"[{op} BACKWARD] Tensor2 '{t2_name}' grad before: {tensor2.grad.data() if tensor2.grad is not None else None}")

    @staticmethod
    def _log_after_grad(op, tensor1, tensor2) -> None:
        t1_name = tensor1._debug_name() or "unnamed_tensor1"
        t2_name = tensor2._debug_name() or "unnamed_tensor2"
        print(f"[{op} BACKWARD] Tensor1 '{t1_name}' grad after: {tensor1.grad.data() if tensor1.grad is not None else None}")
        print(f"[{op} BACKWARD] Tensor2 '{t2_name}' grad after: {tensor2.grad.data() if tensor2.grad is not None else None}")