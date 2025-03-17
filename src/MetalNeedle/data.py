from .device import DeviceManager
from .util import ShapeUtils


class TensorData:
    def __init__(self, data: list[int], dtype, device, debug_name=None):
        tensor, self.operations = DeviceManager.set_dtype_tensor(dtype, device)
        _shape = ShapeUtils.get_shape(data)
        self.raw_tensor = ShapeUtils.create_data_struct(tensor, data, _shape)
        self._debug_name = debug_name

    @staticmethod
    def create(raw_tensor, operations, debug_name=None):
        result = TensorData.__new__(TensorData)
        result.raw_tensor = raw_tensor
        result.operations = operations
        result._debug_name = debug_name
        return result

    def clone(self):
        new_raw_tensor = self.raw_tensor.initialize(self.data(), self.shape())
        result = TensorData.__new__(TensorData)
        result.raw_tensor = new_raw_tensor
        # result.tensor = self.tensor
        result.operations = self.operations
        result._debug_name = self._debug_name + "_clone" if self._debug_name is not None else "tensor_clone"
        return result

    def _init(self, tensor):
        self.raw_tensor = tensor

    def shape(self):
        return self.raw_tensor.shape

    def set_shape(self, shape):
        self.raw_tensor.shape = shape

    def stride(self):
        return self.raw_tensor.stride

    def set_stride(self, shape):
        self.raw_tensor.stride = shape

    def data(self):
        return self.raw_tensor.data

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.raw_tensor.data[index]
        elif isinstance(index, list):
            return self.mult_dim_to_flat_index(index)
        raise TypeError("index must be a list or int")

    def mult_dim_to_flat_index(self, idx):
        return self.raw_tensor.mult_dim_to_flat_index(idx)

    def __add__(self, value):
        if DeviceManager.is_tensor(value):
            raw_tensor = self.operations.ewise_add(self.raw_tensor, value)
            return TensorData.create(raw_tensor, self.operations)
        elif isinstance(value, TensorData):
            raw_tensor = self.operations.ewise_add(self.raw_tensor, value.raw_tensor)
            return TensorData.create(raw_tensor, self.operations)
        elif isinstance(value, (int, float)):
            raw_tensor = self.operations.scalar_add(self.raw_tensor, value)
            return TensorData.create(raw_tensor, self.operations)
        raise TypeError("invalid add")

    def __sub__(self, value):
        if DeviceManager.is_tensor(value):
            raw_tensor = self.operations.ewise_sub(self.raw_tensor, value)
            return TensorData.create(raw_tensor, self.operations)
        elif isinstance(value, TensorData):
            raw_tensor = self.operations.ewise_sub(self.raw_tensor, value.raw_tensor)
            return TensorData.create(raw_tensor, self.operations)
        elif isinstance(value, (int, float)):
            raw_tensor = self.operations.scalar_sub(self.raw_tensor, value)
            return TensorData.create(raw_tensor, self.operations)
        raise TypeError("invalid sub")

    def __mul__(self, value):
        if DeviceManager.is_tensor(value):
            raw_tensor = self.operations.ewise_mul(self.raw_tensor, value)
            return TensorData.create(raw_tensor, self.operations)
        elif isinstance(value, TensorData):
            raw_tensor = self.operations.ewise_mul(self.raw_tensor, value.raw_tensor)
            return TensorData.create(raw_tensor, self.operations)
        elif isinstance(value, (int, float)):
            raw_tensor = self.operations.scalar_mul(self.raw_tensor, value)
            return TensorData.create(raw_tensor, self.operations)
        raise TypeError("invalid mul")

    def __truediv__(self, value):
        if DeviceManager.is_tensor(value):
            raw_tensor = self.operations.ewise_div(self.raw_tensor, value)
            return TensorData.create(raw_tensor, self.operations)
        elif isinstance(value, TensorData):
            raw_tensor = self.operations.ewise_div(self.raw_tensor, value.raw_tensor)
            return TensorData.create(raw_tensor, self.operations)
        elif isinstance(value, (int, float)):
            raw_tensor = self.operations.scalar_div(self.raw_tensor, value)
            return TensorData.create(raw_tensor, self.operations)
        raise TypeError("invalid div")

    def __pow__(self, value):
        if DeviceManager.is_tensor(value):
            raw_tensor = self.operations.ewise_exp(self.raw_tensor, value)
            return TensorData.create(raw_tensor, self.operations)
        elif isinstance(value, TensorData):
            raw_tensor = self.operations.ewise_exp(self.raw_tensor, value.raw_tensor)
            return TensorData.create(raw_tensor, self.operations)
        elif isinstance(value, (int, float)):
            raw_tensor = self.operations.scalar_exp(self.raw_tensor, value)
            return TensorData.create(raw_tensor, self.operations)
        raise TypeError("invalid exp")

    def log(self):
        raw_tensor = self.operations.log(self.raw_tensor)
        return TensorData.create(raw_tensor, self.operations)

    def __matmul__(self, value):
        raw_tensor = self.operations.mat_mul(self.raw_tensor, value.raw_tensor)
        return TensorData.create(raw_tensor, self.operations)

    def broadcast(self, new_shape: list[int]):
        current_shape = self.shape()[:]
        new_stride = []
        if len(current_shape) > len(new_shape):
            raise ValueError("Cannot broadcast to smaller dimensions")

        for i in range(1, len(new_shape) + 1):
            curr_dim = current_shape[-i] if i <= len(current_shape) else 1
            target_dim = new_shape[-i]

            if curr_dim == 1 and target_dim > 1:
                new_stride.insert(0, 0)
            elif curr_dim == target_dim:
                stride_item = self.stride()[-i]
                new_stride.insert(0,  stride_item)
            else:
                raise ValueError(f"Incompatible broadcast: {curr_dim} to {target_dim}")

        self.raw_tensor.stride = new_stride
        self.raw_tensor.shape = new_shape

    def sum(self, axes: list[int], keepDims):
        _data = self.operations.sum(self.raw_tensor, axes, keepDims)
        return TensorData.create(_data, self.operations)

    @property
    def T(self):
        return self.transpose()

    # clone in order to not be destructive
    def transpose(self):
        result = self.clone()
        result.raw_tensor.swap(0, 1)
        return result

    def swap(self, axis1, axis2):
        if axis1 >= len(self.shape()) or axis2 >= len(self.shape()):
            raise ValueError("axes for swap out of range")
        self.raw_tensor.swap(axis1, axis2)
        return self

    def ones_like(self):
        ones_data = self.raw_tensor.create(self.shape(), 1)
        _data = self.raw_tensor.initialize(ones_data, self.shape())
        return TensorData.create(_data, self.operations, 'ones')

    def __str__(self):
        shape = ', '.join(str(x) for x in self.shape())
        return f"<{self.__class__.__module__}.{self.__class__.__name__}> (size: [{shape}])"
