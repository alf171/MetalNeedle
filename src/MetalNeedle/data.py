from .device import DeviceManager
from .util import ShapeUtils


class TensorData:
    def __init__(self, data: list[int], _tensor, _operations):
        _shape = ShapeUtils.get_shape(data)
        self.rawTensor = ShapeUtils.create_data_struct(_tensor, data, _shape)
        self.operations = _operations

    @staticmethod
    def create(rawTensor, operations):
        result = TensorData.__new__(TensorData)
        result.rawTensor = rawTensor
        result.operations = operations
        return result

    def _init(self, tensor):
        self.rawTensor = tensor

    def shape(self):
        return self.rawTensor.shape

    def setShape(self, shape):
        self.rawTensor.shape = shape

    def stride(self):
        return self.rawTensor.stride

    def setStride(self, shape):
        self.rawTensor.stride = shape

    def __getitem__(self, index):
        return self.rawTensor.data[index]

    def mult_dim_to_flat_index(self, idx):
        return self.rawTensor.mult_dim_to_flat_index(idx)

    def __add__(self, value):
        if DeviceManager.is_tensor(value):
            rawTensor = self.operations.ewise_add(self.rawTensor, value)
            return TensorData.create(rawTensor, self.operations)
        elif isinstance(value, TensorData):
            rawTensor = self.operations.ewise_add(self.rawTensor, value.rawTensor)
            return TensorData.create(rawTensor, self.operations)
        elif isinstance(value, (int, float)):
            rawTensor = self.operations.scalar_add(self.rawTensor, value)
            return TensorData.create(rawTensor, self.operations)
        raise TypeError("invalid add")

    def __sub__(self, value):
        if DeviceManager.is_tensor(value):
            rawTensor = self.operations.ewise_sub(self.rawTensor, value)
            return TensorData.create(rawTensor, self.operations)
        elif isinstance(value, TensorData):
            rawTensor = self.operations.ewise_sub(self.rawTensor, value.rawTensor)
            return TensorData.create(rawTensor, self.operations)
        elif isinstance(value, (int, float)):
            rawTensor = self.operations.scalar_sub(self.rawTensor, value)
            return TensorData.create(rawTensor, self.operations)
        raise TypeError("invalid sub")

    def __mul__(self, value):
        if DeviceManager.is_tensor(value):
            rawTensor = self.operations.ewise_mul(self.rawTensor, value)
            return TensorData.create(rawTensor, self.operations)
        elif isinstance(value, TensorData):
            rawTensor = self.operations.ewise_mul(self.rawTensor, value.rawTensor)
            return TensorData.create(rawTensor, self.operations)
        elif isinstance(value, (int, float)):
            rawTensor = self.operations.scalar_mul(self.rawTensor, value)
            return TensorData.create(rawTensor, self.operations)
        raise TypeError("invalid mul")

    def __truediv__(self, value):
        if DeviceManager.is_tensor(value):
            rawTensor = self.operations.ewise_div(self.rawTensor, value)
            return TensorData.create(rawTensor, self.operations)
        elif isinstance(value, TensorData):
            rawTensor = self.operations.ewise_div(self.rawTensor, value.rawTensor)
            return TensorData.create(rawTensor, self.operations)
        elif isinstance(value, (int, float)):
            rawTensor = self.operations.scalar_div(self.rawTensor, value)
            return TensorData.create(rawTensor, self.operations)
        raise TypeError("invalid div")

    def __pow__(self, value):
        if DeviceManager.is_tensor(value):
            rawTensor = self.operations.ewise_exp(self.rawTensor, value)
            return TensorData.create(rawTensor, self.operations)
        elif isinstance(value, TensorData):
            rawTensor = self.operations.ewise_exp(self.rawTensor, value.rawTensor)
            return TensorData.create(rawTensor, self.operations)
        elif isinstance(value, (int, float)):
            rawTensor = self.operations.scalar_exp(self.rawTensor, value)
            return TensorData.create(rawTensor, self.operations)
        raise TypeError("invalid exp")

    def __matmul__(self, value):
        rawTensor = self.operations.mat_mul(self.rawTensor, value.rawTensor)
        return TensorData.create(rawTensor, self.operations)

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

        self.rawTensor.stride = new_stride
        self.rawTensor.shape = new_shape

    def sum(self, axes: list[int]):
        rawTensor = self.operations.sum(self.rawTensor, axes)
        return TensorData.create(rawTensor, self.operations)

    def swap(self, axis1, axis2):
        self.rawTensor.shape[axis1], self.rawTensor.shape[axis2] = self.rawTensor.shape[axis2], self.rawTensor.shape[axis1]
        self.rawTensor.stride[axis1], self.rawTensor.stride[axis2] = self.rawTensor.stride[axis2], self.rawTensor.stride[axis1]
