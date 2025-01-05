from .util import ShapeUtils
from .ops import TensorOperations
from .device import DeviceManager

class Tensor:
    def __init__(self, data, shape = None, device="cpu", dtype="int32", requires_grad=False):
        self.device = device
        self.dtype = dtype
        self._tensor, self._operations = DeviceManager.set_dtype_tensor(dtype, self.device)
        self.shape = ShapeUtils.get_shape(data, device) if shape is None else shape
        # this is actually a tensor so a little confusing
        self._data = ShapeUtils.create_data_struct(self._tensor, data, self.shape)
        self.ops = TensorOperations(self._operations)

        # TODO: add automatic differentiation
        self.node = None
        self.requires_grad = requires_grad

    # allows caller to pipe in _data (C++ version of data)
    def _init(self, data, device = None, shape = None, dtype = None, ops = None):
        result = Tensor.__new__(Tensor)
        result.device = self.device if device is None else device
        result.shape = self.shape if shape is None else shape
        result.dtype = self.dtype if dtype is None else dtype
        result.ops = self.ops if ops is None else TensorOperations(ops)
        result._data = data
        return result

    # TODO: support partial slicing with
    def __getitem__(self, multi_dim_index):
        if not isinstance(multi_dim_index, (list, tuple)):
            multi_dim_index = [multi_dim_index]

        if len(multi_dim_index) > len(self.shape):
            raise ValueError("index shape exceeds tensor's dimensions")

        res = []
        for (index, dim) in zip(multi_dim_index, self.shape):
            if isinstance(index, slice):
                res.append(range(*index.indices(dim)))
            elif isinstance(index, int):
                if not 0 <= index < dim:
                    raise ValueError(f"index {index} out of bounds on dim {dim}")
                res.append([index])
            else:
                raise TypeError(f"Unsupported index data type: {type(index)}")

        indices = ShapeUtils.cartesian_product(res)

        def get_item(idx):
            flat_index = self._data.mult_dim_to_flat_index(idx)
            return self._data.data[flat_index]

        if len(indices) == 1:
            return get_item(indices[0])

        # TODO: reshape back to original shape
        return [get_item(index) for index in indices]

    # TODO: use keep dims and broadcast shape
    # note: this is a destructive operation
    def sum(self, axes, keepdim = False):
        if axes is None or axes == []:
            raise TypeError(f"Axes Cant be Null")
        if not isinstance(axes, list):
            axes = [axes]

        self._data = self.ops.sum(self._data, axes)
        self.shape = self._data.shape

    def transpose(self, axis1=0, axis2=1):
        self.shape[axis1], self.shape[axis2] = self.shape[axis2], self.shape[axis1]
        self._data.shape[axis1], self._data.shape[axis2] = self._data.shape[axis2], self._data.shape[axis1]
        self._data.stride[axis1], self._data.stride[axis2] = self._data.stride[axis2], self._data.stride[axis1]

    def reshape(self, shape):
        if ShapeUtils.product(shape) != ShapeUtils.product(self.shape):
            raise TypeError(f"original dimension ({self.shape}) product != proposed ({shape})")

        new_stride = []
        acc = 1
        for size in reversed(shape):
            new_stride.insert(0, acc)
            acc *= size

        self.shape = shape
        self._data.stride = new_stride
        self._data.shape = shape

    def __add__(self, other):
        if isinstance(other, Tensor):
            return self._init(self.ops.add(self._data, other._data))
        elif isinstance(other, (int, float)):
            return self._init(self.ops.scalar_add(self._data, other))
        raise TypeError(f"Can't add Tensor of type {self.dtype} with {type(other)}")

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return self._init(self.ops.sub(self._data, other._data))
        elif isinstance(other, (int, float)):
            return self._init(self.ops.scalar_sub(self._data, other))
        raise TypeError(f"Can't subtract Tensor of type {self.dtype} with {type(other)}")

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return self._init(self.ops.mul(self._data, other._data))
        elif isinstance(other, (int, float)):
            return self._init(self.ops.scalar_mul(self._data, other))
        raise TypeError(f"Can't multiply Tensor of type {self.dtype} with {type(other)}")

    def __div__(self, other):
        if isinstance(other, Tensor):
            return self._init(self.ops.div(self._data, other._data))
        elif isinstance(other, (int, float)):
            return self._init(self.ops.scalar_div(self._data, other))
        raise TypeError(f"Can't divide Tensor of type {self.dtype} with {type(other)}")

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            new_shape = self.shape[:-1] + other.shape[1:];
            return self._init(self.ops.matmul(self._data, other._data), new_shape)
        raise TypeError(f"Can't divide Tensor of type {self.dtype} with {type(other)}")

    def __exp__(self, other):
        if isinstance(other, Tensor):
            return self._init(self.ops.exp(self._data, other._data))
        if isinstance(other, (int, float)):
            return self._init(self.ops.scalar_exp(self._data, other._data))
        raise TypeError(f"Can't exponentiate Tensor of type {self.dtype} with {type(other)}")

    # print shape and dtype in addition to default stuff
    def __str__(self):
        shape = ', '.join(str(x) for x in self.shape)
        return f"<{self.__class__.__module__}.{self.__class__.__name__}> (size: [{shape}], dtype={self.dtype})"

