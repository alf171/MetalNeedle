from __future__ import annotations

import itertools
from typing import List, Optional, Union, TypeVar

from .device import DeviceManager, TensorDtypes, TensorDevices
from .util import TensorUtils

T = TypeVar("T", bound=Union[int, float])


class TensorData:
    def __init__(self, data: List[T], dtype: str, device: str, debug_name=None):
        self._dtype = TensorDtypes(dtype)
        self._device = TensorDevices(device)
        self.raw_tensor = DeviceManager.get_tensor(self.dtype, self._device)
        self.operations = DeviceManager.get_backend(self.dtype, self._device)
        _shape = TensorUtils.get_shape(data)
        self.raw_tensor.initialize(TensorUtils.flatten(data), _shape)
        self._debug_name = debug_name

    def _create(self, raw_tensor, debug_name=None) -> TensorData:
        result = TensorData.__new__(TensorData)
        result.raw_tensor = raw_tensor
        result.operations = self.operations
        result._dtype = self.dtype
        result._device = self.device
        result._debug_name = debug_name
        return result

    @staticmethod
    def create(
        raw_tensor,
        operations,
        dtype: TensorDtypes,
        device: TensorDevices,
        debug_name=None,
    ) -> TensorData:
        result = TensorData.__new__(TensorData)
        result.raw_tensor = raw_tensor
        result.operations = operations
        result._dtype = dtype
        result._device = device
        result._debug_name = debug_name
        return result

    @staticmethod
    def load_from_buffer(
        data: Union[bytes, list[T]],
        shape: list[int] | tuple[int],
        dtype: str,
        device: str,
        debug_name: Union[str, None],
        normalize: Optional[float] = None,
    ) -> TensorData:
        result = TensorData.__new__(TensorData)
        result._dtype = TensorDtypes(dtype)
        result._device = TensorDevices(device)
        result.raw_tensor = DeviceManager.get_tensor(result._dtype, result._device)
        result.operations = DeviceManager.get_backend(result._dtype, result._device)
        if isinstance(data, bytes):
            if normalize is None:
                normalize = 255.0
            result.raw_tensor.initialize(data, shape, normalize)
        else:
            result.raw_tensor.initialize(data, shape)
        result._debug_name = debug_name
        return result

    def view(self, debug_name = None) -> TensorData:
        ranges = [(0, dim) for dim in self.shape()]
        raw_tensor = self.operations.slice(self.raw_tensor, ranges)
        return TensorData.create(
            raw_tensor,
            self.operations,
            self._dtype,
            self._device,
            debug_name if debug_name is not None else self._debug_name,
        )

    def clone(self) -> TensorData:
        new_raw_tensor = self.raw_tensor.create(self.data(), self.shape())
        result = TensorData.__new__(TensorData)
        result.raw_tensor = new_raw_tensor
        result.operations = self.operations
        result._dtype = self.dtype
        result._device = self.device
        result._debug_name = (
            self._debug_name + "_clone"
            if self._debug_name is not None
            else "tensor_clone"
        )
        return result

    def shape(self) -> List[int]:
        return self.raw_tensor.shape

    def set_shape(self, shape) -> None:
        self.raw_tensor.set_metadata(shape, self.raw_tensor.stride, self.raw_tensor.offset)

    def stride(self) -> List[int]:
        return self.raw_tensor.stride

    def set_stride(self, shape) -> None:
        self.raw_tensor.set_metadata(self.raw_tensor.shape, shape, self.raw_tensor.offset)

    def data(self) -> List[T]:
        return self.raw_tensor.data()

    def offset(self) -> int:
        return self.raw_tensor.offset

    def tensor_count(self) -> int:
        return self.raw_tensor.get_tensor_count()

    @property
    def dtype(self) -> TensorDtypes:
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype: TensorDtypes) -> None:
        """
        Convert tensor to different data type
        """
        if new_dtype != self._dtype:
            # new_raw_tensor = self.raw_tensor.as_type(new_dtype.value)
            if new_dtype == TensorDtypes.float:
                self.raw_tensor = self.raw_tensor.as_float()
                self.operations = DeviceManager.get_backend(new_dtype, self._device)
                self._dtype = new_dtype
            raise NotImplemented(
                f"dtype conversion from {self._dtype} to {new_dtype} not supported"
            )

    @property
    def device(self) -> TensorDevices:
        return self._device

    def __neg__(self):
        return TensorData.__mul__(self, -1)

    def __setitem__(self, key, value):
        if isinstance(key, int):
            key = [key]
        self.raw_tensor.set_item(key, value)

    def __getitem__(self, index) -> TensorData:
        if isinstance(index, list):
            raw_tensor = self.operations.slice(self.raw_tensor, index)
            return TensorData.create(
                raw_tensor, self.operations, self._dtype, self._device
            )
        elif isinstance(index, int):
            return self.get_single_item([index])

        raise TypeError("index must be a list or int")

    def get_single_item(self, index: List[int]) -> T:
        compact_index = 0
        for axis, size in enumerate(self.shape()):
            stride = 1
            for next_size in self.shape()[axis + 1 :]:
                stride *= next_size
            compact_index += index[axis] * stride
        return self.data()[compact_index]

    def __add__(self, value) -> TensorData:
        if DeviceManager.is_tensor(value):
            raw_tensor = self.operations.ewise_add(self.raw_tensor, value)
            return self._create(raw_tensor)
        elif isinstance(value, TensorData):
            raw_tensor = self.operations.ewise_add(self.raw_tensor, value.raw_tensor)
            return self._create(raw_tensor)
        elif isinstance(value, (int, float)):
            raw_tensor = self.operations.scalar_add(self.raw_tensor, value)
            return self._create(raw_tensor)
        raise TypeError("invalid add")

    def __sub__(self, value) -> TensorData:
        if DeviceManager.is_tensor(value):
            raw_tensor = self.operations.ewise_sub(self.raw_tensor, value)
            return self._create(raw_tensor)
        elif isinstance(value, TensorData):
            raw_tensor = self.operations.ewise_sub(self.raw_tensor, value.raw_tensor)
            return self._create(raw_tensor)
        elif isinstance(value, (int, float)):
            raw_tensor = self.operations.scalar_sub(self.raw_tensor, value)
            return self._create(raw_tensor)
        raise TypeError("invalid sub")

    def __mul__(self, value) -> TensorData:
        if DeviceManager.is_tensor(value):
            raw_tensor = self.operations.ewise_mul(self.raw_tensor, value)
            return self._create(raw_tensor)
        elif isinstance(value, TensorData):
            raw_tensor = self.operations.ewise_mul(self.raw_tensor, value.raw_tensor)
            return self._create(raw_tensor)
        elif isinstance(value, (int, float)):
            raw_tensor = self.operations.scalar_mul(self.raw_tensor, value)
            return self._create(raw_tensor)
        raise TypeError(f"invalid mul self type: {type(self)} value type {type(value)}")

    def __truediv__(self, value) -> TensorData:
        if DeviceManager.is_tensor(value):
            raw_tensor = self.operations.ewise_div(self.raw_tensor, value)
            return self._create(raw_tensor)
        elif isinstance(value, TensorData):
            raw_tensor = self.operations.ewise_div(self.raw_tensor, value.raw_tensor)
            return self._create(raw_tensor)
        elif isinstance(value, (int, float)):
            raw_tensor = self.operations.scalar_div(self.raw_tensor, value)
            return self._create(raw_tensor)
        raise TypeError("invalid div")

    def __pow__(self, value) -> TensorData:
        if DeviceManager.is_tensor(value):
            raw_tensor = self.operations.ewise_pow(self.raw_tensor, value)
            return self._create(raw_tensor)
        elif isinstance(value, TensorData):
            raw_tensor = self.operations.ewise_pow(self.raw_tensor, value.raw_tensor)
            return self._create(raw_tensor)
        elif isinstance(value, (int, float)):
            raw_tensor = self.operations.scalar_pow(self.raw_tensor, value)
            return self._create(raw_tensor)
        raise TypeError("invalid exp")

    def exp(self):
        raw_tensor = self.operations.exp(self.raw_tensor)
        return self._create(raw_tensor)

    def log(self) -> TensorData:
        raw_tensor = self.operations.log(self.raw_tensor)
        return self._create(raw_tensor)

    def __matmul__(self, value) -> TensorData:
        raw_tensor = self.operations.mat_mul(self.raw_tensor, value.raw_tensor)
        return self._create(raw_tensor)

    def broadcast(self, new_shape: List[int]) -> TensorData:
        result = self.view()
        current_shape = result.shape()[:]
        current_stride = result.stride()[:]
        new_stride = []

        while len(current_shape) < len(new_shape):
            current_shape.insert(0, 1)
            current_stride.insert(0, 0)

        for i in range(len(new_shape)):
            curr_dim = current_shape[i]
            target_dim = new_shape[i]

            if curr_dim == 1 and target_dim > 1:
                new_stride.append(0)
            elif curr_dim == target_dim:
                new_stride.append(current_stride[i])
            else:
                raise ValueError(f"Incompatible broadcast: {curr_dim} to {target_dim}")

        result.set_stride(new_stride)
        result.set_shape(new_shape)
        return result

    def reshape(self, new_shape: List[int]) -> TensorData:
        if TensorUtils.product(new_shape) != TensorUtils.product(self.shape()):
            raise TypeError(
                f"original dimension ({self.shape()}) product != proposed ({new_shape})"
            )

        new_stride = []
        acc = 1
        for size in reversed(new_shape):
            new_stride.insert(0, acc)
            acc *= size

        result = self.view()
        result.set_shape(new_shape)
        result.set_stride(new_stride)
        return result

    def sum(self, axes: List[int], keep_dims=False) -> TensorData:
        raw_tensor = self.operations.sum(self.raw_tensor, axes, keep_dims)
        return self._create(raw_tensor)

    @property
    def T(self) -> TensorData:
        return self.transpose()

    def transpose(self) -> TensorData:
        result = self.view()
        result.raw_tensor.swap(0, 1)
        return result

    def swap(self, axis1: int, axis2: int) -> TensorData:
        if axis1 >= len(self.shape()) or axis2 >= len(self.shape()):
            raise ValueError("axes for swap out of range")
        result = self.view()
        result.raw_tensor.swap(axis1, axis2)
        return result

    def ones_like(self, new_shape: Union[List[int], None] = None) -> TensorData:
        new_shape = new_shape if new_shape is not None else self.shape()
        ones_data = self.raw_tensor.fill(new_shape, 1)
        return self._create(ones_data, "ones")

    def zeros_like(self, new_shape=None) -> TensorData:
        new_shape = new_shape if new_shape is not None else self.shape()
        zero_data = self.raw_tensor.fill(new_shape, 0)
        return self._create(zero_data, "zeros")

    def maximum(self, value) -> TensorData:
        if DeviceManager.is_tensor(value):
            raw_tensor = self.operations.ewise_max(self.raw_tensor, value)
            return self._create(raw_tensor)
        elif isinstance(value, TensorData):
            raw_tensor = self.operations.ewise_max(self.raw_tensor, value.raw_tensor)
            return self._create(raw_tensor)
        elif isinstance(value, (int, float)):
            raw_tensor = self.operations.scalar_max(self.raw_tensor, value)
            return self._create(raw_tensor)
        raise TypeError(f"invalid maximum type {type(value)}")

    def minimum(self, value) -> TensorData:
        if DeviceManager.is_tensor(value):
            raw_tensor = self.operations.ewise_min(self.raw_tensor, value)
            return self._create(raw_tensor)
        elif isinstance(value, TensorData):
            raw_tensor = self.operations.ewise_min(self.raw_tensor, value.raw_tensor)
            return self._create(raw_tensor)
        elif isinstance(value, (int, float)):
            raw_tensor = self.operations.scalar_min(self.raw_tensor, value)
            return self._create(raw_tensor)
        raise TypeError(f"invalid minimum type {type(value)}")

    def max(self, axes: list[int], keep_dims: bool) -> TensorData:
        raw_tensor = self.raw_tensor.max(axes, keep_dims)
        return self._create(raw_tensor)

    def __gt__(self, other: Union[TensorData, T]) -> TensorData:
        if isinstance(other, TensorData):
            raw_tensor = self.operations.greater_than_tensor(
                self.raw_tensor, other.raw_tensor
            )
            return self._create(raw_tensor)
        elif isinstance(other, (int, float)):
            raw_tensor = self.operations.greater_than_scalar(self.raw_tensor, other)
            return self._create(raw_tensor)
        raise TypeError(f"other is not a supported type: {type(other)}")

    def __ge__(self, other: Union[TensorData, T]) -> TensorData:
        if isinstance(other, TensorData):
            raw_tensor = self.operations.greater_equal_tensor(
                self.raw_tensor, other.raw_tensor
            )
            return self._create(raw_tensor)
        elif isinstance(other, (int, float)):
            raw_tensor = self.operations.greater_equal_scalar(self.raw_tensor, other)
            return self._create(raw_tensor)
        raise TypeError(f"other is not a supported type: {type(other)}")

    def __lt__(self, other: Union[TensorData, T]) -> TensorData:
        if isinstance(other, TensorData):
            raw_tensor = self.operations.less_than_tensor(
                self.raw_tensor, other.raw_tensor
            )
            return self._create(raw_tensor)
        elif isinstance(other, (int, float)):
            raw_tensor = self.operations.less_than_scalar(self.raw_tensor, other)
            return self._create(raw_tensor)
        raise TypeError(f"other is not a supported type: {type(other)}")

    def __le__(self, other: Union[TensorData, T]) -> TensorData:
        if isinstance(other, TensorData):
            raw_tensor = self.operations.less_equal_tensor(
                self.raw_tensor, other.raw_tensor
            )
            return self._create(raw_tensor)
        elif isinstance(other, (int, float)):
            raw_tensor = self.operations.less_equal_scalar(self.raw_tensor, other)
            return self._create(raw_tensor)
        raise TypeError(f"other is not a supported type: {type(other)}")

    def __eq__(self, other: Union[TensorData, T]) -> TensorData:
        if isinstance(other, TensorData):
            raw_tensor = self.operations.equal_tensor(self.raw_tensor, other.raw_tensor)
            return self._create(raw_tensor)
        elif isinstance(other, (int, float)):
            raw_tensor = self.operations.equal_scalar(self.raw_tensor, other)
            return self._create(raw_tensor)
        raise TypeError(f"other is not a supported type: {type(other)}")

    def __ne__(self, other: Union[TensorData, T]) -> TensorData:
        if isinstance(other, TensorData):
            raw_tensor = self.operations.not_equal_tensor(
                self.raw_tensor, other.raw_tensor
            )
            return self._create(raw_tensor)
        elif isinstance(other, (int, float)):
            raw_tensor = self.operations.not_equal_scalar(self.raw_tensor, other)
            return self._create(raw_tensor)
        raise TypeError(f"other is not a supported type: {type(other)}")

    def clip(
        self, lower: Union[TensorData, T], upper: Union[TensorData, T]
    ) -> TensorData:
        lower_is_scalar = isinstance(lower, (int, float))
        upper_is_scalar = isinstance(upper, (int, float))
        lower_is_tensor = isinstance(lower, TensorData)
        upper_is_tensor = isinstance(upper, TensorData)
        if lower_is_scalar and upper_is_scalar:
            raw_tensor = self.operations.clip_scalar_scalar(
                self.raw_tensor, lower, upper
            )
            return self._create(raw_tensor)
        elif lower_is_scalar and upper_is_tensor:
            raw_tensor = self.operations.clip_scalar_tensor(
                self.raw_tensor, lower, upper.raw_tensor
            )
            return self._create(raw_tensor)
        elif lower_is_tensor and upper_is_scalar:
            raw_tensor = self.operations.clip_tensor_scalar(
                self.raw_tensor, lower.raw_tensor, upper
            )
            return self._create(raw_tensor)
        elif lower_is_tensor and upper_is_tensor:
            raw_tensor = self.operations.clip_tensor_tensor(
                self.raw_tensor, lower.raw_tensor, upper.raw_tensor
            )
            return self._create(raw_tensor)

        raise TypeError(
            f"[clip] lower: {type(lower)} and upper: {type(upper)} is not a supported type"
        )

    def compact(self) -> None:
        self.raw_tensor.compact()

    def _zero(self) -> None:
        """Zero out tensor data"""
        self.raw_tensor.fill(self.shape(), 0)

    def __str__(self) -> str:
        shape = ", ".join(str(x) for x in self.shape())
        return (
            f"<{self.__class__.__module__}.{self.__class__.__name__}> (size: [{shape}])"
        )
