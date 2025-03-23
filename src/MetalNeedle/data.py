from .device import DeviceManager
from .util import ShapeUtils


class TensorData:
    def __init__(self, data: list[int], dtype, device, debug_name=None):
        self.raw_tensor, self.operations = DeviceManager.set_dtype_tensor(dtype, device)
        _shape = ShapeUtils.get_shape(data)
        ShapeUtils.create_data_struct(self.raw_tensor, data, _shape)
        self._debug_name = debug_name

    @staticmethod
    def create(raw_tensor, operations, debug_name=None):
        result = TensorData.__new__(TensorData)
        result.raw_tensor = raw_tensor
        result.operations = operations
        result._debug_name = debug_name
        return result

    def clone(self):
        new_raw_tensor = self.raw_tensor.create(self.data(), self.shape())
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
    def reshape(self, new_shape):
        if ShapeUtils.product(new_shape) != ShapeUtils.product(self.shape()):
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

    def sum(self, axes: list[int], keep_dims):
        _data = self.operations.sum(self.raw_tensor, axes, keep_dims)
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
        ones_data = self.raw_tensor.fill(self.shape(), 1)
        _data = self.raw_tensor.create(ones_data, self.shape())
        return TensorData.create(_data, self.operations, 'ones')

    def __str__(self):
        shape = ', '.join(str(x) for x in self.shape())
        return f"<{self.__class__.__module__}.{self.__class__.__name__}> (size: [{shape}])"
