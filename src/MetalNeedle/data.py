from .device import DeviceManager
from .util import ShapeUtils


class TensorData:
    def __init__(self, data: list[int], _tensor, _operations):
        _shape = ShapeUtils.get_shape(data)
        self.tensor = ShapeUtils.create_data_struct(_tensor, data, _shape)
        self.operations = _operations

    @staticmethod
    def create(data, operations):
        result = TensorData.__new__(TensorData)
        result.tensor = data
        result.operations = operations
        return result

    def _init(self, tensor):
        self.tensor = tensor

    def shape(self):
        return self.tensor.shape

    def setShape(self, shape):
        self.tensor.shape = shape

    def stride(self):
        return self.tensor.stride

    def setStride(self, shape):
        self.tensor.stride = shape

    def __getitem__(self, index):
        return self.tensor.data[index]

    def mult_dim_to_flat_index(self, idx):
        return self.tensor.mult_dim_to_flat_index(idx)

    def __add__(self, value):
        if DeviceManager.is_tensor(value):
            return self.operations.ewise_add(self.tensor, value)
        elif isinstance(value, (int, float)):
            return self.operations.scalar_add(self.tensor, value)
        raise TypeError("invalid add")

    def __sub__(self, value):
        if DeviceManager.is_tensor(value):
            return self.operations.ewise_sub(self.tensor, value)
        elif isinstance(value, (int, float)):
            return self.operations.scalar_sub(self.tensor, value)
        raise TypeError("invalid sub")

    def __mul__(self, value):
        if DeviceManager.is_tensor(value):
            return self.operations.ewise_mul(self.tensor, value)
        elif isinstance(value, (int, float)):
            return self.operations.scalar_mul(self.tensor, value)
        raise TypeError("invalid mul")

    def __truediv__(self, value):
        if DeviceManager.is_tensor(value):
            return self.operations.ewise_div(self.tensor, value)
        elif isinstance(value, (int, float)):
            return self.operations.scalar_div(self.tensor, value)
        raise TypeError("invalid div")

    def __pow__(self, value):
        if DeviceManager.is_tensor(value):
            return self.operations.ewise_exp(self.tensor, value)
        elif isinstance(value, (int, float)):
            return self.operations.scalar_exp(self.tensor, value)
        raise TypeError("invalid exp")

    def __matmul__(self, value):
        return self.operations.mat_mul(self.tensor, value)

    def sum(self, axes):
        return self.operations.sum(self.tensor, axes)
