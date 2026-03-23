from __future__ import annotations

import array
import os
import struct
import uuid
from typing import List, Union, TypeVar, Optional

from .device import (
    DTYPE_TO_ARRAY_ENCODE,
    TensorDevices,
    TensorDtypes,
    DTYPE_TO_SIZE_ENCODE,
)
from .util import TensorUtils
from .ops import TensorOperations
from .data import TensorData

T = TypeVar("T", bound=Union[int, float])

class Tensor:
    def __init__(
        self,
        data: list[T],
        device="cpu",
        dtype="int32",
        requires_grad=False,
        debug_name=None,
    ):
        """
        default initialization when size is unknown
        like when data is provided ex. mn.Tensor([1,2,3])
        """
        self.tensor_data: TensorData = TensorData(data, dtype, device, debug_name)
        self.requires_grad: bool = requires_grad
        self.grad: Union[TensorData, None] = None
        self.grad_fn = None

    @staticmethod
    def create(
        raw_tensor: List[T],
        device: str,
        dtype: str,
        operations,
        requires_grad=False,
        debug_name=None,
    ) -> Tensor:
        """
        used externally to create a new tensor or create a copy
        """
        result = Tensor.__new__(Tensor)
        result.tensor_data = TensorData.create(
            raw_tensor,
            operations,
            TensorDtypes(dtype),
            TensorDevices(device),
            debug_name,
        )
        result.requires_grad = requires_grad
        result.grad = None
        result.grad_fn = None
        return result

    @staticmethod
    def load_from_buffer(
        buffer: Union[bytes, list[T]],
        shape: list[int] | tuple[int],
        device: str = "cpu",
        dtype: str = "int32",
        requires_grad=False,
        debug_name=None,
    ) -> Tensor:
        """
        Useful for outside data loads since we already know the shape and have flat data
        """
        result = Tensor.__new__(Tensor)
        result.tensor_data = TensorData.load_from_buffer(
            buffer, shape, dtype, device, debug_name
        )
        result.requires_grad = requires_grad
        result.grad = None
        result.grad_fn = None
        return result

    @staticmethod
    def load_from_file(
        file: str, requires_grad=False, debug_name=None, offset=0
    ) -> Tensor:
        """
        Load a Tensor from a file. Shape, Dtype, and Device are stored in the file
        while requires_grad and debug_name are not and must thus be set on load call.
        """
        try:
            with open(file, "rb") as f:
                f.seek(offset)

                shape_rank = struct.unpack("I", f.read(4))[0]
                shape = struct.unpack(f"{shape_rank}I", f.read(4 * shape_rank))

                dtype_rank = struct.unpack("I", f.read(4))[0]
                dtype = f.read(dtype_rank).decode("utf-8")

                device_rank = struct.unpack("I", f.read(4))[0]
                device = f.read(device_rank).decode("utf-8")

                element_size = DTYPE_TO_SIZE_ENCODE[TensorDtypes(dtype)]
                buffer = f.read(TensorUtils.product(shape) * element_size)
                print(f"[load] shape={shape}, dtype={dtype}, read_bytes={len(buffer)}")
                return Tensor.load_from_buffer(
                    buffer, shape, device, dtype, requires_grad, debug_name
                )
        except (OSError, IOError) as e:
            raise ValueError(f"Failed to open or read file '{file}': {e}")

    def clone(self) -> Tensor:
        """
        Creates a deep copy of the tensor with completely independent memory.
        """
        res = Tensor.create(
            raw_tensor=self.data(),
            device=self.device.value,
            dtype=self.dtype.value,
            operations=self.tensor_data.operations,
            requires_grad=self.requires_grad,
            debug_name=f"{self.debug_name()}_clone"
            if self.debug_name
            else "cloned_tensor",
        )
        res.grad_fn = lambda grad: self.backward(grad)
        return res

    def _init(self, data: TensorData, _grad_fn=None, requires_grad: bool | None = None, debug_name=None) -> Tensor:
        """
        Used internally to create a new tensor from TensorData
        """
        result = Tensor.__new__(Tensor)
        result.tensor_data = data
        result.tensor_data._debug_name = debug_name
        result.requires_grad = self.requires_grad if requires_grad is None else requires_grad
        result.grad_fn = _grad_fn
        result.grad = None
        return result

    def save_to_file(
        self, file_name: Optional[str] = None, offset: int = 0, directory="."
    ) -> None:
        """
        Save a Tensors metadata and weights to a file
        """

        if file_name is None:
            file_name = f"{self.debug_name()}_{uuid.uuid4().hex[:8]}"

        path = os.path.join(directory, file_name)

        dtype_str = self.dtype.value.encode("utf-8")
        device_str = self.device.value.encode("utf-8")
        data_bytes = self.to_bytes()
        print(
            f"[save] shape={self.shape()}, dtype={self.dtype.value}, bytes={len(data_bytes)}"
        )
        with open(path, "wb") as f:
            if offset:
                f.seek(offset)

            # write shape
            f.write(struct.pack("I", len(self.shape())))
            f.write(struct.pack(f"{len(self.shape())}I", *self.shape()))
            # write dtype
            f.write(struct.pack("I", len(dtype_str)))
            f.write(dtype_str)
            # write device
            f.write(struct.pack("I", len(device_str)))
            f.write(device_str)
            # write data
            f.write(data_bytes)

    def shape(self) -> List[int]:
        return self.tensor_data.shape()

    def data(self) -> List[T]:
        return self.tensor_data.data()

    @property
    def device(self) -> TensorDevices:
        return self.tensor_data.device

    @property
    def dtype(self) -> TensorDtypes:
        return self.tensor_data.dtype

    def tensor_count(self) -> int:
        return self.tensor_data.tensor_count()

    def debug_name(self) -> str:
        if self.tensor_data._debug_name:
            return self.tensor_data._debug_name
        return ""

    def numel(self) -> int:
        return TensorUtils.product(self.shape())

    def to_bytes(self) -> bytes:
        fmt = DTYPE_TO_ARRAY_ENCODE[self.dtype]
        if fmt is None:
            raise ValueError(f"dtype {self.dtype} is not recognized")
        return array.array(fmt, self.data()).tobytes()

    def __neg__(self) -> Tensor:
        debug_name = (
            f"negative_{self.debug_name()}"
            if self.debug_name() is not None
            else "negative_tensor"
        )
        return self.__mul__(-1, debug_name)

    def __setitem__(self, key: int, value: T, debug_name=None) -> None:
        self.tensor_data[key] = value

    def __getitem__(
        self, multi_dim_index: Union[int, List[int], tuple[int, ...]], debug_name=None
    ) -> Union[Tensor, T]:
        if isinstance(multi_dim_index, int):
            multi_dim_index = [multi_dim_index]

        if len(multi_dim_index) > len(self.tensor_data.shape()):
            raise ValueError("index shape exceeds tensor's dimensions")

        all_are_indices = True

        ranges = []
        for index, dim in zip(multi_dim_index, self.tensor_data.shape()):
            if isinstance(index, slice):
                start, stop, _ = index.indices(dim)
                ranges.append((start, stop))
                all_are_indices = False
            elif isinstance(index, int):
                if not 0 <= index < dim:
                    raise ValueError(f"index {index} out of bounds on dim {dim}")
                ranges.append((index, index + 1))
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
    def T(self, debug_name=None) -> Tensor:
        return self.transpose(debug_name)

    def transpose(self, debug_name=None) -> Tensor:
        (tensor_data, _grad_fn) = TensorOperations.swap(self, 0, 1)
        return self._init(tensor_data, _grad_fn, debug_name)

    def swap(self, axis1: int, axis2: int) -> None:
        # if axis is negative, index opposite direction
        # consider moving this code lower down the stack
        if axis2 < 0:
            axis2 = len(self.shape()) + axis2
        TensorOperations.swap(self, axis1, axis2)

    def reshape(self, new_shape: list[int], debug_name=None) -> Tensor:
        (tensor_data, _grad_fn) = TensorOperations.reshape(self, new_shape)
        res = self._init(tensor_data, _grad_fn, debug_name)
        return res

    def __add__(self, other: Union[Tensor, T], debug_name=None) -> Tensor:
        if isinstance(other, Tensor):
            if not TensorUtils.can_broadcast(self.shape(), other.shape()):
                raise ValueError(
                    f"[ADD] cant broadcast {self.shape} with {other.shape}"
                )

            if self.shape() == other.shape():
                (tensor_data, _grad_fn) = TensorOperations.add(self, other)
                res = self._init(tensor_data, _grad_fn, self.requires_grad or other.requires_grad, debug_name)
                return res
            else:
                broadcast_other = other.broadcast(self.shape())
                (tensor_data, _grad_fn) = TensorOperations.add(self, broadcast_other)
                res = self._init(tensor_data, _grad_fn, self.requires_grad or other.requires_grad, debug_name)
                return res

        elif isinstance(other, (int, float)):
            (tensor_data, _grad_fn) = TensorOperations.scalar_add(self, other)
            res = self._init(tensor_data, _grad_fn)
            return res
        raise TypeError(f"Can't add Tensor of type {self.dtype} with {type(other)}")

    def __sub__(self, other: Union[Tensor, T]) -> Tensor:
        if isinstance(other, Tensor):
            (tensor_data, _grad_fn) = TensorOperations.sub(self, other)
            res = self._init(tensor_data, _grad_fn, self.requires_grad or other.requires_grad)
            return res
        elif isinstance(other, (int, float)):
            (tensor_data, _grad_fn) = TensorOperations.scalar_sub(self, other)
            res = self._init(tensor_data, _grad_fn, self.requires_grad)
            return res
        raise TypeError(
            f"Can't subtract Tensor of type {self.dtype} with {type(other)}"
        )

    def __mul__(self, other: Union[Tensor, T], debug_name=None) -> Tensor:
        if isinstance(other, Tensor):
            (tensor_data, _grad_fn) = TensorOperations.mul(self, other)
            res = self._init(tensor_data, _grad_fn, self.requires_grad or other.requires_grad, debug_name)
            return res
        elif isinstance(other, (int, float)):
            (tensor_data, _grad_fn) = TensorOperations.scalar_mul(self, other)
            res = self._init(tensor_data, _grad_fn, self.requires_grad, debug_name)
            return res
        raise TypeError(
            f"Can't multiply Tensor of type {self.dtype} with {type(other)}"
        )

    def __truediv__(self, other: Union[Tensor, T]) -> Tensor:
        if isinstance(other, Tensor):
            (tensor_data, _grad_fn) = TensorOperations.div(self, other)
            res = self._init(tensor_data, _grad_fn, self.requires_grad or other.requires_grad)
            return res
        elif isinstance(other, (int, float)):
            (tensor_data, _grad_fn) = TensorOperations.scalar_div(self, other)
            res = self._init(tensor_data, _grad_fn, self.requires_grad)
            return res
        raise TypeError(f"Can't divide Tensor of type {self.dtype} with {type(other)}")

    def __pow__(self, other: Union[Tensor, T]) -> Tensor:
        if isinstance(other, Tensor):
            (tensor_data, _grad_fn) = TensorOperations.pow(self, other)
            res = self._init(tensor_data, _grad_fn, self.requires_grad or other.requires_grad)
            return res
        if isinstance(other, (int, float)):
            (tensor_data, _grad_fn) = TensorOperations.scalar_pow(self, other)
            res = self._init(tensor_data, _grad_fn, self.requires_grad)
            return res
        raise TypeError(
            f"Can't exponentiate Tensor of type {self.dtype} with {type(other)}"
        )

    def exp(self):
        (tensor_data, _grad_fn) = TensorOperations.exp(self)
        res = self._init(tensor_data, _grad_fn)
        return res

    def log(self) -> Tensor:
        (tensor_data, _grad_fn) = TensorOperations.scalar_log(self)
        res = self._init(tensor_data, _grad_fn)
        return res

    def __matmul__(self, other: Union[Tensor, T]) -> Tensor:
        if isinstance(other, Tensor):
            (tensor_data, _grad_fn) = TensorOperations.matmul(self, other)
            res = self._init(tensor_data, _grad_fn, self.requires_grad or other.requires_grad)
            return res

        raise TypeError(f"other is of type {type(other)} not Tensor")

    def maximum(self, other: Union[Tensor, T]) -> Tensor:
        """
        Maximum of a tensor or a scalar value
        """
        if isinstance(other, Tensor):
            (tensor_data, _grad_fn) = TensorOperations.maximum(self, other)
            res = self._init(tensor_data, _grad_fn, self.requires_grad or other.requires_grad)
            return res
        elif isinstance(other, (int, float)):
            (tensor_data, _grad_fn) = TensorOperations.scalar_maximum(self, other)
            res = self._init(tensor_data, _grad_fn, self.requires_grad)
            return res

        raise TypeError(f"other is of type {type(other)} not Tensor")

    def max(self, axes: List[int] | int = [], keep_dims=False) -> Tensor:
        """
        Maximum value with a tensor or axis
        """
        normalized_axes = TensorUtils.normalize_axes(axes, self.shape(), "max")

        tensor_data, _grad_fn = TensorOperations.max(self, normalized_axes, keep_dims)
        res = self._init(tensor_data, _grad_fn)
        return res

    def minimum(self, other: Union[Tensor, T]) -> Tensor:
        """ """
        if isinstance(other, Tensor):
            (tensor_data, _grad_fn) = TensorOperations.minimum(self, other)
            res = self._init(tensor_data, _grad_fn, self.requires_grad or other.requires_grad)
            return res
        elif isinstance(other, (int, float)):
            (tensor_data, _grad_fn) = TensorOperations.scalar_minimum(self, other)
            return self._init(tensor_data, _grad_fn, self.requires_grad)

        raise TypeError(f"[minimum] {type(other)} is not a supported type")

    def clip(self, lower: Union[Tensor, T], upper: Union[Tensor, T]) -> Tensor:
        """
        Clip a tensor in between min and max.
        A combination of maximum and minimum
        """
        lower_is_scalar = isinstance(lower, (int, float))
        upper_is_scalar = isinstance(upper, (int, float))
        lower_is_tensor = isinstance(lower, Tensor)
        upper_is_tensor = isinstance(upper, Tensor)

        if lower_is_scalar and upper_is_scalar:
            (tensor_data, _grad_fn) = TensorOperations.clip_scalar_scalar(
                self, lower, upper
            )
            return self._init(tensor_data, _grad_fn)
        elif lower_is_scalar and upper_is_tensor:
            (tensor_data, _grad_fn) = TensorOperations.clip_scalar_tensor(
                self, lower, upper
            )
            return self._init(tensor_data, _grad_fn)
        elif lower_is_tensor and upper_is_scalar:
            (tensor_data, _grad_fn) = TensorOperations.clip_tensor_scalar(
                self, lower, upper
            )
            return self._init(tensor_data, _grad_fn)
        elif lower_is_tensor and upper_is_tensor:
            (tensor_data, _grad_fn) = TensorOperations.clip_tensor_tensor(
                self, lower, upper
            )
            return self._init(tensor_data, _grad_fn)

        raise TypeError(
            f"[clip] lower: {type(lower)} and upper: {type(upper)} is not a supported type"
        )

    def mean(self, axes=None, keep_dims=False) -> Tensor:
        """Mean of a tensor reducing across axes"""
        reduce_sum = self.sum(axes, keep_dims)
        if axes is None:
            divisor = TensorUtils.product(self.tensor_data.shape())
        elif isinstance(axes, int):
            divisor = self.shape()[axes]
        elif isinstance(axes, list):
            divisor = TensorUtils.product(self.shape()[ax] for ax in axes)

        return reduce_sum / divisor

    def sum(self, axes=None, keep_dims=False) -> Tensor:
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

        # HACK: compact gradient since operations don't respect shape,stride,offset of non copmact tensors
        grad.compact()
        if self.grad is None:
            self.grad = grad
        else:
            # make sure to accumulate grads
            self.grad.compact()
            assert self.grad is not None
            self.grad = TensorData.__add__(self.grad, grad)

        if self.grad is not None and self.debug_name() is not None:
            self.grad._debug_name = self.debug_name() + "_grad"

        if self.grad_fn is not None:
            self.grad_fn(grad)

    def one_hot(self, num_classes):
        """
        one_hot(y)_i = 1[i = y]
        tensor of shape goes from shape -> [...shape, num_classes]
        """
        new_shape = self.shape() + [num_classes]
        one_hot = self.tensor_data.zeros_like(new_shape)
        one_hot_tensor = self._init(one_hot, None, debug_name="one_hot")
        for i in range(self.shape()[0]):
            scalar = int(self[i])
            one_hot_tensor[i, scalar] = 1

        return one_hot_tensor

    def compact(self):
        """
        compactify the tensor we are looking at
        """
        self.tensor_data.compact()

    def __gt__(self, other: Union[Tensor, T]) -> Tensor:
        if isinstance(other, Tensor):
            tensor_data, _grad_fn = TensorOperations.greater_than_tensor(self, other)
            return self._init(tensor_data, _grad_fn)
        elif isinstance(other, (int, float)):
            tensor_data, _grad_fn = TensorOperations.greater_than_scalar(self, other)
            return self._init(tensor_data, _grad_fn)
        raise TypeError(f"other is not a supported type: {type(other)}")

    def __ge__(self, other: Union[Tensor, T]) -> Tensor:
        if isinstance(other, Tensor):
            tensor_data, _grad_fn = TensorOperations.greater_than_or_eq_tensor(
                self, other
            )
            return self._init(tensor_data, _grad_fn)
        elif isinstance(other, (int, float)):
            tensor_data, _grad_fn = TensorOperations.greater_than_or_eq_scalar(
                self, other
            )
            return self._init(tensor_data, _grad_fn)
        raise TypeError(f"other is not a supported type: {type(other)}")

    def __lt__(self, other: Union[Tensor, T]) -> Tensor:
        if isinstance(other, Tensor):
            tensor_data, _grad_fn = TensorOperations.less_than_tensor(self, other)
            return self._init(tensor_data, _grad_fn)
        elif isinstance(other, (int, float)):
            tensor_data, _grad_fn = TensorOperations.less_than_scalar(self, other)
            return self._init(tensor_data, _grad_fn)
        raise TypeError(f"other is not a supported type: {type(other)}")

    def __le__(self, other: Union[Tensor, T]) -> Tensor:
        if isinstance(other, Tensor):
            tensor_data, _grad_fn = TensorOperations.less_than_or_eq_tensor(self, other)
            return self._init(tensor_data, _grad_fn)
        elif isinstance(other, (int, float)):
            tensor_data, _grad_fn = TensorOperations.less_than_or_eq_scalar(self, other)
            return self._init(tensor_data, _grad_fn)
        raise TypeError(f"other is not a supported type: {type(other)}")

    def __eq__(self, other: Union[Tensor, T]) -> Tensor:
        if isinstance(other, Tensor):
            tensor_data, _grad_fn = TensorOperations.eq_tensor(self, other)
            return self._init(tensor_data, _grad_fn)
        elif isinstance(other, (int, float)):
            tensor_data, _grad_fn = TensorOperations.eq_scalar(self, other)
            return self._init(tensor_data, _grad_fn)
        raise TypeError(f"other is not a supported type: {type(other)}")

    def __ne__(self, other: Union[Tensor, T]) -> Tensor:
        if isinstance(other, Tensor):
            tensor_data, _grad_fn = TensorOperations.neq_tensor(self, other)
            return self._init(tensor_data, _grad_fn)
        elif isinstance(other, (int, float)):
            tensor_data, _grad_fn = TensorOperations.neq_scalar(self, other)
            return self._init(tensor_data, _grad_fn)
        raise TypeError(f"other is not a supported type: {type(other)}")

    def __str__(self) -> str:
        shape = ", ".join(str(x) for x in self.shape())
        return f"<{self.__class__.__module__}.{self.__class__.__name__}> (size: [{shape}], dtype={self.dtype})"
