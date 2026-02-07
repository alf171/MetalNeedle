from typing import TypeVar, Union

from MetalNeedle.data import TensorData

T = TypeVar("T", bound=Union[int, float])

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
            tensor1_grad = tensor2.tensor_data * grad
            tensor1.backward(tensor1_grad)
        if tensor2.requires_grad:
            tensor2_grad = tensor1.tensor_data * grad
            tensor2.backward(tensor2_grad)

    @staticmethod
    def scalar_mul(grad: TensorData, tensor1, value: T) -> None:
        if tensor1.requires_grad:
            tensor1.backward(grad * value)

    @staticmethod
    def div(grad: TensorData, tensor1, tensor2) -> None:
        # da(A/B) = 1/B
        if tensor1.requires_grad:
            tensor1_grad = grad / tensor2.tensor_data
            tensor1.backward(tensor1_grad)
        # db(A/B) = -A/B^2
        if tensor2.requires_grad:
            tensor2_grad = (-grad * tensor1.tensor_data) / (tensor2.tensor_data**2)
            tensor2.backward(tensor2_grad)

    @staticmethod
    def scalar_div(grad: TensorData, tensor1, value: T) -> None:
        if tensor1.requires_grad:
            tensor1.backward(grad / value)

    @staticmethod
    def pow(grad: TensorData, tensor1, tensor2) -> None:
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
    def scalar_pow(grad: TensorData, tensor1, value: T) -> None:
        if tensor1.requires_grad:
            tensor1_grad = grad * (value * tensor1.tensor_data ** (value - 1))
            tensor1.backward(tensor1_grad)

    @staticmethod
    def exp(grad: TensorData, tensor1) -> None:
        if tensor1.requires_grad:
            tensor1_grad = grad * tensor1.tensor_data.exp()
            tensor1.backward(tensor1_grad)

    @staticmethod
    def scalar_log(grad: TensorData, tensor1) -> None:
        if tensor1.requires_grad:
            tensor1.backward(grad / tensor1.tensor_data)

    @staticmethod
    def matmul(grad: TensorData, tensor1, tensor2) -> None:
        if tensor1.requires_grad:
            tensor1_grad = grad @ tensor2.tensor_data.T
            tensor1.backward(tensor1_grad)
        if tensor2.requires_grad:
            tensor2_grad = tensor1.tensor_data.T @ grad
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

    # TODO: impl with where(cond, a, b) to reduce boiler plate
    @staticmethod
    def maximum(grad: TensorData, tensor1, tensor2) -> None:
        # dx(max(x,y)) = {grad if x > y, else 0 }
        if tensor1.requires_grad:
            raise NotImplementedError("maximum back not implemented")
        # dy(max(x,y)) = {grad if x < y, else 0 }
        if tensor2.requires_grad:
            raise NotImplementedError("maximum back not implemented")

    @staticmethod
    def minimum(grad: TensorData, tensor1, tensor2) -> None:
        if tensor1.requires_grad:
            raise NotImplementedError("minimum back not implemented")
        if tensor2.requires_grad:
            raise NotImplementedError("minimum back not implemented")

    @staticmethod
    def max(grad: TensorData, tensor1) -> None:
        if tensor1.requires_grad:
            # During forward, save the argmax locations (where the max happened).
            # During backward, create a mask (1s where input == max, 0s elsewhere).
            # Multiply incoming grad by the mask.
            # Then broadcast it back to the original input shape if needed.
            raise NotImplementedError("max back not implemented")

    @staticmethod
    def clip_scalar_scalar(grad: TensorData, tensor1) -> None:
        if tensor1.requires_grad:
            raise NotImplementedError("clip back not implemented")

    @staticmethod
    def clip_tensor_scalar(grad: TensorData, tensor, lower_tensor) -> None:
        if tensor.requires_grad:
            raise NotImplementedError("clip back not implemented")
        if lower_tensor.requires_grad:
            raise NotImplementedError("clip back not implemented")

    @staticmethod
    def clip_scalar_tensor(grad: TensorData, tensor, upper_tensor) -> None:
        if tensor.requires_grad:
            raise NotImplementedError("clip back not implemented")
        if upper_tensor.requires_grad:
            raise NotImplementedError("clip back not implemented")

    @staticmethod
    def clip_tensor_tensor(
        grad: TensorData, tensor, lower_tensor, upper_tensor
    ) -> None:
        if tensor.requires_grad:
            raise NotImplementedError("clip back not implemented")
        if lower_tensor.requires_grad:
            raise NotImplementedError("clip back not implemented")
        if upper_tensor.requires_grad:
            raise NotImplementedError("clip back not implemented")

    @staticmethod
    def greater_than_tensor(grad: TensorData, tensor1, tensor2) -> None:
        if tensor1.requires_grad:
            mask = tensor1.tensor_data > tensor2.tensor_data
            tensor1.backward(mask * grad)

    @staticmethod
    def greater_than_scalar(grad: TensorData, tensor, val: T) -> None:
        if tensor.requires_grad:
            mask = tensor.tensor_data > val
            tensor.backward(mask * grad)

    @staticmethod
    def greater_than_or_eq_tensor(grad: TensorData, tensor1, tensor2) -> None:
        if tensor1.requires_grad:
            tensor1_grad = grad >= tensor2.tensor_data
            tensor1.backward(tensor1_grad)
        if tensor2.requires_grad:
            tensor2_grad = grad >= tensor1.tensor_data
            tensor2.backward(tensor2_grad)

    @staticmethod
    def greater_than_or_eq_scalar(grad: TensorData, tensor, val: T) -> None:
        if tensor.requires_grad:
            tensor_grad = grad >= val
            tensor.backward(tensor_grad)

    @staticmethod
    def less_than_tensor(grad: TensorData, tensor1, tensor2) -> None:
        if tensor1.requires_grad:
            tensor1_grad = grad < tensor2.tensor_data
            tensor1.backward(tensor1_grad)
        if tensor2.requires_grad:
            tensor2_grad = grad < tensor1.tensor_data
            tensor2.backward(tensor2_grad)

    @staticmethod
    def less_than_scalar(grad: TensorData, tensor, val) -> None:
        if tensor.requires_grad:
            tensor_grad = grad < val
            tensor.backward(tensor_grad)

    @staticmethod
    def less_than_or_eq_tensor(grad: TensorData, tensor1, tensor2) -> None:
        if tensor1.requires_grad:
            tensor1_grad = grad <= tensor2.tensor_data
            tensor1.backward(tensor1_grad)
        if tensor2.requires_grad:
            tensor2_grad = grad <= tensor1.tensor_data
            tensor2.backward(tensor2_grad)

    @staticmethod
    def less_than_or_eq_scalar(grad: TensorData, tensor, val: T) -> None:
        if tensor.requires_grad:
            tensor_grad = grad <= val
            tensor.backward(tensor_grad)

    @staticmethod
    def eq_tensor(grad: TensorData, tensor1, tensor2) -> None:
        if tensor1.requires_grad:
            tensor1_grad = grad == tensor2.tensor_data
            tensor1.backward(tensor1_grad)
        if tensor2.requires_grad:
            tensor2_grad = grad == tensor1.tensor_data
            tensor2.backward(tensor2_grad)

    @staticmethod
    def eq_scalar(grad: TensorData, tensor, val: T) -> None:
        if tensor.requires_grad:
            tensor_grad = grad == val
            tensor.backward(tensor_grad)

    @staticmethod
    def neq_tensor(grad: TensorData, tensor1, tensor2) -> None:
        if tensor1.requires_grad:
            tensor1_grad = grad != tensor2.tensor_data
            tensor1.backward(tensor1_grad)
        if tensor2.requires_grad:
            tensor2_grad = grad != tensor1.tensor_data
            tensor2.backward(tensor2_grad)

    @staticmethod
    def neq_scalar(grad: TensorData, tensor, val: T) -> None:
        if tensor.requires_grad:
            tensor_grad = grad != val
            tensor.backward(tensor_grad)

    @staticmethod
    def _log_before_grad(op: str, grad, tensor1=None, tensor2=None) -> None:
        # Log information about the incoming gradient
        print(
            f"[{op} BACKWARD] Gradient shape: {grad.shape() if hasattr(grad, 'shape') else 'scalar'}"
        )
        print(
            f"[{op} BACKWARD] Gradient value: {grad.data() if hasattr(grad, 'data') else grad}"
        )

        # Log information about the tensors being added
        if tensor1 is not None:
            t1_name = tensor1._debug_name() or "unnamed_tensor1"
            print(
                f"[{op} BACKWARD] Tensor1 '{t1_name}' shape: {tensor1.shape()}, requires_grad: {tensor1.requires_grad}"
            )
        if tensor2 is not None:
            t2_name = tensor2._debug_name() or "unnamed_tensor2"
            print(
                f"[{op} BACKWARD] Tensor2 '{t2_name}' shape: {tensor2.shape()}, requires_grad: {tensor2.requires_grad}"
            )

        # Log the current gradient values of both tensors before update
        if tensor1 is not None:
            t1_name = tensor1._debug_name() or "unnamed_tensor1"
            print(
                f"[{op} BACKWARD] Tensor1 '{t1_name}' grad before: {tensor1.grad.data() if tensor1.grad is not None else None}"
            )
        if tensor2 is not None:
            t2_name = tensor2._debug_name() or "unnamed_tensor2"
            print(
                f"[{op} BACKWARD] Tensor2 '{t2_name}' grad before: {tensor2.grad.data() if tensor2.grad is not None else None}"
            )

    @staticmethod
    def _log_after_grad(op: str, tensor1, tensor2) -> None:
        t1_name = tensor1._debug_name() or "unnamed_tensor1"
        t2_name = tensor2._debug_name() or "unnamed_tensor2"
        print(
            f"[{op} BACKWARD] Tensor1 '{t1_name}' grad after: {tensor1.grad.data() if tensor1.grad is not None else None}"
        )
        print(
            f"[{op} BACKWARD] Tensor2 '{t2_name}' grad after: {tensor2.grad.data() if tensor2.grad is not None else None}"
        )
