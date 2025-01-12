from .util import ShapeUtils
from .ops import TensorOperations
from .device import DeviceManager

class Tensor:
    def __init__(self, data, shape = None, device="cpu", dtype="int32", requires_grad=False):
        self.device = device
        self.dtype = dtype
        self._tensor, self._operations = DeviceManager.set_dtype_tensor(dtype, self.device)
        # this is actually a tensor so a little confusing
        _shape = ShapeUtils.get_shape(data, device) if shape is None else shape
        self._data = ShapeUtils.create_data_struct(self._tensor, data, _shape)
        self.ops = TensorOperations(self._operations)

        # autograd related
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None

    # allows caller to pipe in _data (C++ version of data)
    def _init(self, data, device = None, shape = None, dtype = None, ops = None, requires_grad = None):
        result = Tensor.__new__(Tensor)
        result.device = self.device if device is None else device
        result.dtype = self.dtype if dtype is None else dtype
        result.ops = self.ops if ops is None else TensorOperations(ops)
        result._data = data
        result.requires_grad = self.requires_grad if requires_grad is None else requires_grad
        result.grad = None
        return result

    # TODO: support partial slicing with
    def __getitem__(self, multi_dim_index):
        if not isinstance(multi_dim_index, (list, tuple)):
            multi_dim_index = [multi_dim_index]

        if len(multi_dim_index) > len(self._data.shape):
            raise ValueError("index shape exceeds tensor's dimensions")

        res = []
        for (index, dim) in zip(multi_dim_index, self._data.shape):
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
        # could also consider returning back a Tensor
        return [get_item(index) for index in indices]

    # TODO: use keep dims and broadcast shape
    # note: this is a destructive operation
    def sum(self, axes, keepdim = False):
        if axes is None or axes == []:
            raise TypeError(f"Axes Cant be Null")
        if not isinstance(axes, list):
            axes = [axes]

        self._data = self.ops.sum(self, axes)

    def transpose(self, axis1=0, axis2=1):
        self._data.shape[axis1], self._data.shape[axis2] = self._data.shape[axis2], self._data.shape[axis1]
        self._data.stride[axis1], self._data.stride[axis2] = self._data.stride[axis2], self._data.stride[axis1]

    def reshape(self, shape):
        if ShapeUtils.product(shape) != ShapeUtils.product(self._data.shape):
            raise TypeError(f"original dimension ({self._data.shape}) product != proposed ({shape})")

        new_stride = []
        acc = 1
        for size in reversed(shape):
            new_stride.insert(0, acc)
            acc *= size

        self._data.stride = new_stride
        self._data.shape = shape

    def __add__(self, other):
        if isinstance(other, Tensor):
            (_data, _backward) = self.ops.add(self, other)
            res = self._init(_data)
            res.grad_fn = _backward
            res.parents = [self, other]
            return res
        elif isinstance(other, (int, float)):
            (_data, _backward) = self.ops.scalar_add(self, other)
            res = self._init(_data)
            res.grad_fn = _backward
            res.parents = [self, other]
            return res
        raise TypeError(f"Can't add Tensor of type {self.dtype} with {type(other)}")

    def __sub__(self, other):
        if isinstance(other, Tensor):
            (_data, _backward) = self.ops.sub(self, other)
            res = self._init(_data)
            res.parents = [self, other]
            res.grad_fn = _backward
            return res
        elif isinstance(other, (int, float)):
            (_data, _backward) = self.ops.scalar_sub(self, other)
            res = self._init(_data)
            res.parents = [self]
            res.grad_fn = _backward
            return res
        raise TypeError(f"Can't subtract Tensor of type {self.dtype} with {type(other)}")

    def __mul__(self, other):
        if isinstance(other, Tensor):
            (_data, _backward) = self.ops.mul(self, other)
            res = self._init(_data)
            res.grad_fn = _backward
            res.parents = [self, other]
            return res
        elif isinstance(other, (int, float)):
            (_data, _backward) = self.ops.scalar_mul(self, other)
            res = self._init(_data)
            res.grad_fn = _backward
            res.parents = [self]
            return res
        raise TypeError(f"Can't multiply Tensor of type {self.dtype} with {type(other)}")

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return self._init(self.ops.div(self, other))
        elif isinstance(other, (int, float)):
            return self._init(self.ops.scalar_div(self, other))
        raise TypeError(f"Can't divide Tensor of type {self.dtype} with {type(other)}")

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            new_shape = self._data.shape[:-1] + other._data.shape[1:]
            return self._init(self.ops.matmul(self, other), new_shape)
        raise TypeError(f"Can't divide Tensor of type {self.dtype} with {type(other)}")

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return self._init(self.ops.exp(self, other))
        if isinstance(other, (int, float)):
            return self._init(self.ops.scalar_exp(self, other))
        raise TypeError(f"Can't exponentiate Tensor of type {self.dtype} with {type(other)}")

    def __str__(self):
        shape = ', '.join(str(x) for x in self._data.shape)
        return f"<{self.__class__.__module__}.{self.__class__.__name__}> (size: [{shape}], dtype={self.dtype})"

