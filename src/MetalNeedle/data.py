from .device import DeviceManager
from .util import ShapeUtils


class TensorData:
    def __init__(self, data: list[int], _tensor, _operations):
        _shape = ShapeUtils.get_shape(data)
        self.rawTensor = ShapeUtils.create_data_struct(_tensor, data, _shape)
        self.operations = _operations

    @staticmethod
    def create(data, operations):
        result = TensorData.__new__(TensorData)
        result.rawTensor = data
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
            return self.operations.ewise_add(self.rawTensor, value)
        elif isinstance(value, (int, float)):
            return self.operations.scalar_add(self.rawTensor, value)
        raise TypeError("invalid add")

    def __sub__(self, value):
        if DeviceManager.is_tensor(value):
            return self.operations.ewise_sub(self.rawTensor, value)
        elif isinstance(value, (int, float)):
            return self.operations.scalar_sub(self.rawTensor, value)
        raise TypeError("invalid sub")

    def __mul__(self, value):
        if DeviceManager.is_tensor(value):
            return self.operations.ewise_mul(self.rawTensor, value)
        elif isinstance(value, (int, float)):
            return self.operations.scalar_mul(self.rawTensor, value)
        raise TypeError("invalid mul")

    def __truediv__(self, value):
        if DeviceManager.is_tensor(value):
            return self.operations.ewise_div(self.rawTensor, value)
        elif isinstance(value, (int, float)):
            return self.operations.scalar_div(self.rawTensor, value)
        raise TypeError("invalid div")

    def __pow__(self, value):
        if DeviceManager.is_tensor(value):
            return self.operations.ewise_exp(self.rawTensor, value)
        elif isinstance(value, (int, float)):
            return self.operations.scalar_exp(self.rawTensor, value)
        raise TypeError("invalid exp")

    def __matmul__(self, value):
        return self.operations.mat_mul(self.rawTensor, value)

    def sum(self, axes):
        return self.operations.sum(self.rawTensor, axes)
