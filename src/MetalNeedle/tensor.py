from .util import ShapeUtils
from .ops import TensorOperations
from .device import DeviceManager
from .data import TensorData

class Tensor:
    def __init__(self, data, device="cpu", dtype="int32", requires_grad=False):
        self.device: str = device
        self.dtype: str = dtype
        # TODO: try to deprecate these fields
        self._tensor, self._operations = DeviceManager.set_dtype_tensor(self.dtype, self.device)
        self.tensorData: TensorData = TensorData(data, self._tensor, self._operations)
        self.ops = TensorOperations
        # autograd related
        self.parents: list[TensorData] = []
        self.requires_grad: bool = requires_grad
        self.grad = None
        self.grad_fn = None

    @staticmethod
    def create(data, device: str, dtype: str, _tensor, _ops, requires_grad=False):
        result = Tensor.__new__(Tensor)
        result.device = device
        result.dtype = dtype
        result.ops = TensorOperations
        result._tensor = _tensor
        result._operations =  _ops
        result.tensorData = TensorData.create(data, _tensor, _ops)
        result.requires_grad = requires_grad
        result.grad = None
        return result

    def _init(self, data, _backward = None):
        result = Tensor.__new__(Tensor)
        result.device = self.device
        result.dtype = self.dtype
        result.ops = self.ops
        result._operations = self._operations
        result._tensor = self._tensor
        result.tensorData = TensorData.create(data, self._tensor, self._operations)
        result.requires_grad = self.requires_grad
        result.grad_fn = _backward
        result.grad = None
        return result

    def shape(self):
        return self.tensorData.rawTensor.shape

    def data(self):
        return self.tensorData.rawTensor.data

    # TODO: support partial slicing and return a tensor if sum(size) > 1
    def __getitem__(self, multi_dim_index):
        if not isinstance(multi_dim_index, (list, tuple)):
            multi_dim_index = [multi_dim_index]

        if len(multi_dim_index) > len(self.tensorData.shape()):
            raise ValueError("index shape exceeds tensor's dimensions")

        res = []
        for (index, dim) in zip(multi_dim_index, self.tensorData.shape()):
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
            flat_index = self.tensorData.mult_dim_to_flat_index(idx)
            return self.tensorData[flat_index]

        if len(indices) == 1:
            return get_item(indices[0])

        # could also consider returning back a Tensor if sum(tensor.size()) > 1
        return [get_item(index) for index in indices]

    def transpose(self):
        self.swap(0, 1)

    def swap(self, axis1, axis2):
        self.ops.swap(self.tensorData, axis1, axis2)

    def reshape(self, shape: list[int]):
        if ShapeUtils.product(shape) != ShapeUtils.product(self.tensorData.shape()):
            raise TypeError(f"original dimension ({self.tensorData.shape()}) product != proposed ({shape})")

        new_stride = []
        acc = 1
        for size in reversed(shape):
            new_stride.insert(0, acc)
            acc *= size

        self.tensorData.setShape(shape)
        self.tensorData.setStride(new_stride)

    def __add__(self, other):
        # do we not assert shape is same
        if isinstance(other, Tensor):
            (tensorData, _backward) = self.ops.add(self, other)
            res = self._init(tensorData, _backward)
            res.parents = [self.tensorData, other.tensorData]
            return res
        elif isinstance(other, (int, float)):
            (tensorData, _backward) = self.ops.scalar_add(self, other)
            res = self._init(tensorData, _backward)
            res.parents = [self.tensorData]
            return res
        raise TypeError(f"Can't add Tensor of type {self.dtype} with {type(other)}")

    def __sub__(self, other):
        if isinstance(other, Tensor):
            (tensorData, _backward) = self.ops.sub(self, other)
            res = self._init(tensorData, _backward)
            res.parents = [self.tensorData, other.tensorData]
            return res
        elif isinstance(other, (int, float)):
            (tensorData, _backward) = self.ops.scalar_sub(self, other)
            res = self._init(tensorData, _backward)
            res.parents = [self.tensorData]
            return res
        raise TypeError(f"Can't subtract Tensor of type {self.dtype} with {type(other)}")

    def __mul__(self, other):
        if isinstance(other, Tensor):
            (tensorData, _backward) = self.ops.mul(self, other)
            res = self._init(tensorData, _backward)
            res.parents = [self.tensorData, other.tensorData]
            return res
        elif isinstance(other, (int, float)):
            (tensorData, _backward) = self.ops.scalar_mul(self, other)
            res = self._init(tensorData, _backward)
            res.parents = [self.tensorData]
            return res
        raise TypeError(f"Can't multiply Tensor of type {self.dtype} with {type(other)}")

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            (tensorData, _backward) = self.ops.div(self, other)
            res = self._init(tensorData, _backward)
            res.parents = [self.tensorData, other.tensorData]
            return res
        elif isinstance(other, (int, float)):
            (tensorData, _backward) = self.ops.scalar_div(self, other)
            res = self._init(tensorData, _backward)
            res.parents = [self.tensorData]
            return res
        raise TypeError(f"Can't divide Tensor of type {self.dtype} with {type(other)}")

    def __pow__(self, other):
        if isinstance(other, Tensor):
            (tensorData, _backward) = self.ops.exp(self, other)
            res = self._init(tensorData, _backward)
            res.parents = [self.tensorData, other.tensorData]
            return res
        if isinstance(other, (int, float)):
            (tensorData, _backward) = self.ops.scalar_exp(self, other)
            res = self._init(tensorData, _backward)
            res.parents = [self.tensorData]
            return res
        raise TypeError(f"Can't exponentiate Tensor of type {self.dtype} with {type(other)}")

    def log(self):
        (tensorData, _backward) = self.ops.scalar_log(self)
        res = self._init(tensorData, _backward)
        res.parents = [self.tensorData]
        return res

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            (tensorData, _backward) = self.ops.matmul(self, other)
            res = self._init(tensorData, _backward)
            res.parents = [self.tensorData, other.tensorData]
            return res

        raise TypeError(f"Can't divide Tensor of type {self.dtype} with {type(other)}")

    # TODO: use backwards
    # TODO: use keep dims and broadcast shape
    # note: this is a destructive operation
    def sum(self, axes: int or list[int], keepdim = False):
        if axes is None or axes == []:
            raise TypeError(f"Axes Cant be Null")
        if not isinstance(axes, list):
            axes = [axes]

        (tensorData, _backward) = self.ops.sum(self, axes)
        self.tensorData._init(tensorData)
        self.grad_fn = _backward

    def broadcast(self, newShape: list[int]):
        _backward = self.ops.broadcast(self.tensorData, newShape)
        self.grad_fn = _backward

    def __str__(self):
        shape = ', '.join(str(x) for x in self.tensorData.shape())
        return f"<{self.__class__.__module__}.{self.__class__.__name__}> (size: [{shape}], dtype={self.dtype})"
