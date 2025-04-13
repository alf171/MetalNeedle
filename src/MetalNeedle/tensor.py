from __future__ import annotations

import math
from typing import Any, List, Union

from .util import TensorUtils
from .ops import TensorOperations
from .data import TensorData

class Tensor:
    def __init__(self, data: list[Any], device="cpu", dtype="int32", requires_grad=False, debug_name=None):
        """
        default initialization when size is unknown
        like when data is provided ex. mn.Tensor([1,2,3])
        """
        self.device: str = device
        self.dtype: str = dtype
        self.tensor_data: TensorData = TensorData(data, dtype, device, debug_name)
        # autograd related
        self.requires_grad: bool = requires_grad
        self.grad: TensorData or None = None
        self.grad_fn = None

    @staticmethod
    def create(raw_tensor: List[Any], device: str, dtype: str, operations, requires_grad=False, debug_name=None) -> Tensor:
        """
        used externally to create a new tensor or create a copy
        """
        result = Tensor.__new__(Tensor)
        result.device = device
        result.dtype = dtype
        result.tensor_data = TensorData.create(raw_tensor, operations, debug_name)
        result.requires_grad = requires_grad
        result.grad = None
        result.grad_fn = None
        return result

    @staticmethod
    def load(data: List[Any], shape: List[int], device = "cpu", dtype = "int32", requires_grad=False, debug_name=None) -> Tensor:
        result = Tensor.__new__(Tensor)
        result.device = device
        result.dtype = dtype
        result.tensor_data = TensorData.load(data, shape, dtype, device, debug_name)
        result.requires_grad = requires_grad
        result.grad = None
        result.grad_fn = None
        return result

    def clone(self) -> Tensor:
        """
        Creates a deep copy of the tensor with completely independent memory.

        Returns:
            Tensor: A new tensor with identical values but separate memory
        """
        res = Tensor.create(
            raw_tensor=self.data(),
            device=self.device,
            dtype=self.dtype,
            operations=self.tensor_data.operations,
            requires_grad=self.requires_grad,
            debug_name=f"{self.debug_name()}_clone" if self.debug_name else "cloned_tensor"
        )
        res.grad_fn = lambda grad: self.backward(grad)
        return res

    def _init(self, data: TensorData, _grad_fn = None, debug_name = None) -> Tensor:
        """
        Used internally to create a new tensor from TensorData
        """
        result = Tensor.__new__(Tensor)
        result.device = self.device
        result.dtype = self.dtype
        result.tensor_data = data
        result.tensor_data._debug_name = debug_name
        result.requires_grad = self.requires_grad
        result.grad_fn = _grad_fn
        result.grad = None
        return result

    def shape(self) -> list[Any]:
        return self.tensor_data.shape()

    def data(self) -> list[Any]:
        return self.tensor_data.data()

    def tensor_count(self) -> int:
        return self.tensor_data.tensor_count()

    def debug_name(self) -> str or None:
        return self.tensor_data._debug_name

    def __getitem__(self, multi_dim_index: Any, debug_name=None) -> Tensor:
        if isinstance(multi_dim_index, (int, float)):
            multi_dim_index = [multi_dim_index]

        if len(multi_dim_index) > len(self.tensor_data.shape()):
            raise ValueError("index shape exceeds tensor's dimensions")

        all_are_indices = True

        ranges = []
        for (index, dim) in zip(multi_dim_index, self.tensor_data.shape()):
            if isinstance(index, slice):
                start, stop, _ = index.indices(dim)
                ranges.append((start, stop))
                all_are_indices = False
            elif isinstance(index, int):
                if not 0 <= index < dim:
                    raise ValueError(f"index {index} out of bounds on dim {dim}")
                ranges.append((index, index+1))
            else:
                raise TypeError(f"Unsupported index data type: {type(index)}")

        # fetch a single value
        if all_are_indices and len(multi_dim_index) == len(self.shape()):
            return self.tensor_data.get_single_item(multi_dim_index)

        # TODO: this should be moved into operations and then gradient should only
        # be propagated into `gotten` items
        tensor_data = self.tensor_data[ranges]
        return self._init(tensor_data, None, debug_name)


    @property
    def T(self, debug_name = None) -> Tensor:
        return self.transpose(debug_name)

    def transpose(self, debug_name = None) -> Tensor:
        (tensor_data, _grad_fn) = TensorOperations.swap(self, 0, 1)
        return self._init(tensor_data, _grad_fn, debug_name)

    def swap(self, axis1: int, axis2: int) -> Any:
        # if axis is negative, index opposite direction
        # consider moving this code lower down the stack
        if axis2 < 0:
            axis2 = len(self.shape()) + axis2
        TensorOperations.swap(self, axis1, axis2)

    def reshape(self, new_shape: list[int], debug_name = None) -> Tensor:
        (tensor_data, _grad_fn) = TensorOperations.reshape(self, new_shape)
        res = self._init(tensor_data, _grad_fn, debug_name)
        return res

    def __add__(self, other: Any, debug_name=None) -> Tensor:
        if isinstance(other, Tensor):
            if not TensorUtils.can_broadcast(self.shape(), other.shape()):
                raise ValueError(f"[ADD] cant broadcast {self.shape} with {other.shape}")

            if self.shape() == other.shape():
                (tensor_data, _grad_fn) = TensorOperations.add(self, other)
                res = self._init(tensor_data, _grad_fn, debug_name)
                return res
            else:
                broadcast_other = other.broadcast(self.shape())
                (tensor_data, _grad_fn) = TensorOperations.add(self, broadcast_other)
                res = self._init(tensor_data, _grad_fn, debug_name)
                return res

        elif isinstance(other, (int, float)):
            (tensor_data, _grad_fn) = TensorOperations.scalar_add(self, other)
            res = self._init(tensor_data, _grad_fn)
            return res
        raise TypeError(f"Can't add Tensor of type {self.dtype} with {type(other)}")

    def __sub__(self, other: Any) -> Tensor:
        if isinstance(other, Tensor):
            (tensor_data, _grad_fn) = TensorOperations.sub(self, other)
            res = self._init(tensor_data, _grad_fn)
            return res
        elif isinstance(other, (int, float)):
            (tensor_data, _grad_fn) = TensorOperations.scalar_sub(self, other)
            res = self._init(tensor_data, _grad_fn)
            return res
        raise TypeError(f"Can't subtract Tensor of type {self.dtype} with {type(other)}")

    def __mul__(self, other: Any, debug_name=None) -> Tensor:
        if isinstance(other, Tensor):
            (tensor_data, _grad_fn) = TensorOperations.mul(self, other)
            res = self._init(tensor_data, _grad_fn, debug_name)
            return res
        elif isinstance(other, (int, float)):
            (tensor_data, _grad_fn) = TensorOperations.scalar_mul(self, other)
            res = self._init(tensor_data, _grad_fn, debug_name)
            return res
        raise TypeError(f"Can't multiply Tensor of type {self.dtype} with {type(other)}")

    def __truediv__(self, other: Any) -> Tensor:
        if isinstance(other, Tensor):
            (tensor_data, _grad_fn) = TensorOperations.div(self, other)
            res = self._init(tensor_data, _grad_fn)
            return res
        elif isinstance(other, (int, float)):
            (tensor_data, _grad_fn) = TensorOperations.scalar_div(self, other)
            res = self._init(tensor_data, _grad_fn)
            return res
        raise TypeError(f"Can't divide Tensor of type {self.dtype} with {type(other)}")

    def __pow__(self, other: Any) -> Tensor:
        if isinstance(other, Tensor):
            (tensor_data, _grad_fn) = TensorOperations.pow(self, other)
            res = self._init(tensor_data, _grad_fn)
            return res
        if isinstance(other, (int, float)):
            (tensor_data, _grad_fn) = TensorOperations.scalar_pow(self, other)
            res = self._init(tensor_data, _grad_fn)
            return res
        raise TypeError(f"Can't exponentiate Tensor of type {self.dtype} with {type(other)}")

    def exp(self):
        (tensor_data, _grad_fn) = TensorOperations.exp(self)
        res = self._init(tensor_data, _grad_fn)
        return res

    def log(self) -> Tensor:
        (tensor_data, _grad_fn) = TensorOperations.scalar_log(self)
        res = self._init(tensor_data, _grad_fn)
        return res

    def __matmul__(self, other: Any) -> Tensor:
        if isinstance(other, Tensor):
            (tensor_data, _grad_fn) = TensorOperations.matmul(self, other)
            res = self._init(tensor_data, _grad_fn)
            return res

        raise TypeError(f"other is of type {type(other)} not Tensor")

    def maximum(self, other: Union[Tensor, Any]) -> Tensor:
        """
        Maximum of a tensor or a scalar value
        """
        if isinstance(other, Tensor):
            (tensor_data, _grad_fn) = TensorOperations.maximum(self, other)
            res = self._init(tensor_data, _grad_fn)
            return res
        elif isinstance(other, (int, float)):
            (tensor_data, _grad_fn) = TensorOperations.scalar_maximum(self, other)
            res = self._init(tensor_data, _grad_fn)
            return res

        raise TypeError(f"other is of type {type(other)} not Tensor")

    def max(self, axes = None, keep_dims = False) -> Tensor:
        """
        maximum value with a tensor or axis
        """
        normalized_axes = TensorUtils.normalize_axes(axes, self.shape(), "max")

        tensor_data, _grad_fn = TensorOperations.max(self, normalized_axes, keep_dims)
        res = self._init(tensor_data, _grad_fn)
        return res

    def sum(self, axes = None, keep_dims = False) -> Tensor:
        normalized_axes = TensorUtils.normalize_axes(axes, self.shape(), "sum")

        (tensor_data, _grad_fn) = TensorOperations.sum(self, normalized_axes, keep_dims)
        res = self._init(tensor_data, _grad_fn)
        return res

    def broadcast(self, new_shape: list[int]) -> Tensor:
        if self.shape() == new_shape:
            return self.clone()
        tensor_data, _grad_fn = TensorOperations.broadcast(self, new_shape)
        return self._init(tensor_data, _grad_fn)

    def backward(self, grad=None) -> None:
        if grad is None:
            grad = self.tensor_data.ones_like()

        # this is where we do += on the grad so it is not require on the operation
        self.grad = grad if self.grad is None else TensorData.__add__(self.grad, grad)
        if self.debug_name() is not None:
            self.grad._debug_name = self.debug_name() + "_grad"

        if self.grad_fn is not None:
            self.grad_fn(grad)

    def compact(self):
        """
        compactify the tensor we are looking at
        """
        self.tensor_data.compact()

    def __str__(self) -> str:
        shape = ', '.join(str(x) for x in self.shape())
        return f"<{self.__class__.__module__}.{self.__class__.__name__}> (size: [{shape}], dtype={self.dtype})"
