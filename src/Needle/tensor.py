from .util import *
from .ops import TensorOperations
from .device import DeviceManager

# TODO: add automatic differentiation
class Tensor():
    def __init__(self, data, shape = None, device="cpu", dtype="int32"):
        self.device = device
        self.dtype = dtype
        self._tensor, self._operations = DeviceManager.set_dtype_tensor(dtype, self.device)
        self.shape = ShapeUtils.get_shape(data, device) if shape is None else shape
        # C++ version of data
        self._data = ShapeUtils.create_data_struct(self._tensor, data, self.shape)
        self.ops = TensorOperations(self._operations)

    # allows up pipe in _data
    def init(self, data):
        result = Tensor.__new__(Tensor)
        result.device = self.device
        result.shape = self.shape
        result.dtype = self.dtype
        result._data = data
        return result

    def __getitem__(self, multi_dim_index):
        if len(multi_dim_index) > len(self.shape):
            raise ValueError("index shape exceeds tensor's dimensions")

        res = []
        for (index, dim) in zip(multi_dim_index, self.shape):
            if isinstance(index, slice):
                res.append(range(*index.indices(dim)))
            elif isinstance(index, int):
                res.append([index])
            else:
                raise TypeError(f"Unsupported index data type: {type(index)}")

        indices = cartesian_product(res)

        if len(indices) == 1:
            return self.get_item(indices[0])

        # TODO: reshape back to original shape
        return [self.get_item(index) for index in indices]

    def get_item(self, multi_dim_index):
        flat_index = self._data.mult_dim_to_flat_index(multi_dim_index)
        return self._data.data[flat_index]

    def __add__(self, other):
        if isinstance(other, Tensor):
            return self.init(self.ops.add(self._data, other._data))
        elif isinstance(other, (int, float)):
            return self.init(self.ops.scalar_add(self._data, other))
        raise TypeError(f"Can't add Tensor of type {self.dtype} with {type(other)}")

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return self.init(self.ops.sub(self._data, other._data))
        elif isinstance(other, (int, float)):
            return self.init(self.ops.scalar_sub(self._data, other))
        raise TypeError(f"Can't subtract Tensor of type {self.dtype} with {type(other)}")

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return self.init(self.ops.mul(self._data, other._data))
        elif isinstance(other, (int, float)):
            return self.init(self.ops.scalar_mul(self._data, other))
        raise TypeError(f"Can't multiply Tensor of type {self.dtype} with {type(other)}")

    def __div__(self, other):
        if isinstance(other, Tensor):
            return self.init(self.ops.div(self._data, other._data))
        elif isinstance(other, (int, float)):
            return self.init(self.ops.scalar_div(self._data, other))
        raise TypeError(f"Can't divide Tensor of type {self.dtype} with {type(other)}")

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            return self.init(self.ops.matmul(self._data, other._data))
        raise TypeError(f"Can't divide Tensor of type {self.dtype} with {type(other)}")

    # print shape and dtype in addition to default stuff
    def __str__(self):
        shape = ', '.join(str(x) for x in self.shape)
        return f"<{self.__class__.__module__}.{self.__class__.__name__}> (size: [{shape}], dtype={self.dtype})"

