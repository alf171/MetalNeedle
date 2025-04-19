from __future__ import annotations

from typing import Any, List, Union

from .device import DeviceManager, TensorDtypes, TensorDevices
from .util import TensorUtils

class TensorData:
    def __init__(self, data: List[Any], dtype: str, device: str, debug_name=None):
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
    def create(raw_tensor, operations, dtype, device, debug_name=None) -> TensorData:
        result = TensorData.__new__(TensorData)
        result.raw_tensor = raw_tensor
        result.operations = operations
        result._dtype = dtype
        result._device = device
        result._debug_name = debug_name
        return result

    @staticmethod
    def load(data: List[Any], shape: List[int], dtype: str, device: str, debug_name: str) -> TensorData:
        result = TensorData.__new__(TensorData)
        result._dtype = TensorDtypes(dtype)
        result._device = TensorDevices(device)
        result.raw_tensor = DeviceManager.get_tensor(result._dtype, result._device)
        result.operations = DeviceManager.get_backend(result._dtype, result._device)
        result.raw_tensor.initialize(data, shape)
        result._debug_name = debug_name
        return result

    def clone(self) -> TensorData:
        new_raw_tensor = self.raw_tensor.create(self.data(), self.shape())
        result = TensorData.__new__(TensorData)
        result.raw_tensor = new_raw_tensor
        result.operations = self.operations
        result._dtype = self.dtype
        result._device = self.device
        result._debug_name = self._debug_name + "_clone" if self._debug_name is not None else "tensor_clone"
        return result

    def shape(self) -> List[int]:
        return self.raw_tensor.shape

    def set_shape(self, shape) -> None:
        self.raw_tensor.shape = shape

    def stride(self) -> List[int]:
        return self.raw_tensor.stride

    def set_stride(self, shape) -> None:
        self.raw_tensor.stride = shape

    def data(self) -> List[Any]:
        return self.raw_tensor.data()

    def offset(self) -> int:
        return self.raw_tensor.offset

    def tensor_count(self) -> int:
        return self.raw_tensor.get_tensor_count()

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype: TensorDtypes):
        """
        Convert tensor to different data type
        """
        if new_dtype != self._dtype:
            # new_raw_tensor = self.raw_tensor.as_type(new_dtype.value)
            if new_dtype == TensorDtypes.float:
                self.raw_tensor = self.raw_tensor.as_float()
                self.operations = DeviceManager.get_backend(new_dtype, self._device)
                self._dtype = new_dtype
            raise NotImplemented(f"dtype conversion from {self._dtype} to {new_dtype} not supported")

    @property
    def device(self):
        return self._device

    def __setitem__(self, key, value):
        self.raw_tensor.set_item(key, value)

    def __getitem__(self, index) -> TensorData:
        if isinstance(index, list):
            raw_tensor = self.operations.slice(self.raw_tensor, index)
            return TensorData.create(raw_tensor, self.operations, self._dtype, self._device)
        raise TypeError("index must be a list or int")

    def get_single_item(self, index) -> Any:
        index = self.raw_tensor.mult_dim_to_flat_index(index)
        return self.data()[index]

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
        raise TypeError("invalid mul")

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

    def broadcast(self, new_shape: list[int]) -> TensorData:
        result = self.clone()
        current_shape = result.shape()[:]
        new_stride = []
        if len(current_shape) > len(new_shape):
            raise ValueError("Cannot broadcast to smaller dimensions")

        for i in range(1, len(new_shape) + 1):
            curr_dim = current_shape[-i] if i <= len(current_shape) else 1
            target_dim = new_shape[-i]

            if curr_dim == 1 and target_dim > 1:
                new_stride.insert(0, 0)
            elif curr_dim == target_dim:
                stride_item = result.stride()[-i]
                new_stride.insert(0,  stride_item)
            else:
                raise ValueError(f"Incompatible broadcast: {curr_dim} to {target_dim}")

        result.raw_tensor.stride = new_stride
        result.raw_tensor.shape = new_shape
        return result

    # TODO: this implementation assumes the data in contiguous
    # SOLUTION: could call flatten but ideally, we want dont want to have the caller
    # be aware when operation makes the memory non contiguous. For this fix,
    # the c++ class should store whether our array is contiguous or not and then call
    # compact automatically when we chain operations together that make an assumption
    # like this. The question then becomes why not always read memory regardless of the
    # output, the point is we can amortize the cost of flattening our data for better
    # caching properties
    def reshape(self, new_shape) -> TensorData:
        if TensorUtils.product(new_shape) != TensorUtils.product(self.shape()):
            raise TypeError(f"original dimension ({self.shape()}) product != proposed ({new_shape})")

        new_stride = []
        acc = 1
        for size in reversed(new_shape):
            new_stride.insert(0, acc)
            acc *= size

        result = self.clone()
        result.set_shape(new_shape)
        result.set_stride(new_stride)
        return result

    def sum(self, axes: list[int], keep_dims) -> TensorData:
        raw_tensor = self.operations.sum(self.raw_tensor, axes, keep_dims)
        return self._create(raw_tensor)

    @property
    def T(self) -> TensorData:
        return self.transpose()

    # clone in order to not be destructive
    def transpose(self) -> TensorData:
        result = self.clone()
        result.raw_tensor.swap(0, 1)
        return result

    def swap(self, axis1, axis2) -> TensorData:
        if axis1 >= len(self.shape()) or axis2 >= len(self.shape()):
            raise ValueError("axes for swap out of range")
        self.raw_tensor.swap(axis1, axis2)
        return self

    def ones_like(self, new_shape = None) -> TensorData:
        new_shape = new_shape if new_shape is not None else self.shape()
        ones_data = self.raw_tensor.fill(new_shape, 1)
        return self._create(ones_data, 'ones')

    def zeros_like(self, new_shape = None) -> TensorData:
        new_shape = new_shape if new_shape is not None else self.shape()
        zero_data = self.raw_tensor.fill(new_shape, 0)
        return self._create(zero_data, 'zeros')

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

    def max(self, axes: tuple[int], keep_dims: bool) -> Any:
        raw_tensor = self.raw_tensor.max(axes, keep_dims)
        return self._create(raw_tensor)

    def clip(self, lower: Union[TensorData, Any], upper: Union[TensorData, Any]) -> TensorData:
        lower_is_scalar = isinstance(lower, (int, float))
        upper_is_scalar = isinstance(upper, (int, float))
        lower_is_tensor = isinstance(lower, TensorData)
        upper_is_tensor = isinstance(upper, TensorData)
        if lower_is_scalar and upper_is_scalar:
            raw_tensor = self.operations.clip_scalar_scalar(self.raw_tensor, lower, upper)
            return self._create(raw_tensor)
        elif lower_is_scalar and upper_is_tensor:
            raw_tensor = self.operations.clip_scalar_tensor(self.raw_tensor, lower, upper.raw_tensor)
            return self._create(raw_tensor)
        elif lower_is_tensor and upper_is_scalar:
            raw_tensor = self.operations.clip_tensor_scalar(self.raw_tensor, lower.raw_tensor, upper)
            return self._create(raw_tensor)
        elif lower_is_tensor and upper_is_tensor:
            raw_tensor = self.operations.clip_tensor_tensor(self.raw_tensor, lower.raw_tensor, upper.raw_tensor)
            return self._create(raw_tensor)

        raise TypeError(f"[clip] lower: {type(lower)} and upper: {type(upper)} is not a supported type")

    def compact(self) -> None:
        self.raw_tensor.compact()

    def __str__(self) -> str:
        shape = ', '.join(str(x) for x in self.shape())
        return f"<{self.__class__.__module__}.{self.__class__.__name__}> (size: [{shape}])"
