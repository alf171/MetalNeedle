from .util import ShapeUtils
from .ops import TensorOperations
from .device import DeviceManager
from .data import TensorData

class Tensor:
    def __init__(self, data, device="cpu", dtype="int32", requires_grad=False):
        self.device: str = device
        self.dtype: str = dtype
        _tensor, _operations = DeviceManager.set_dtype_tensor(dtype, self.device)
        self.tensorData: TensorData = TensorData(data, _tensor, _operations)
        self.ops = TensorOperations(_operations)
        # autograd related
        self.parents: list[TensorData] = []
        self.requires_grad: bool = requires_grad
        self.grad = None
        self.grad_fn = None

    @staticmethod
    def create(data, device: str, dtype: str, ops, requires_grad=False):
        result = Tensor.__new__(Tensor)
        result.device = device
        result.dtype = dtype
        result.ops = TensorOperations(ops)
        result.tensorData= TensorData.create(data, result.ops)
        result.requires_grad = requires_grad
        result.grad = None
        return result

    def _init(self, data):
        result = Tensor.__new__(Tensor)
        result.device = self.device
        result.dtype = self.dtype
        result.ops = self.ops
        result.tensorData= TensorData.create(data, result.ops)
        result.requires_grad = self.requires_grad
        result.grad = None
        return result

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

        # TODO: reshape back to original shape
        # could also consider returning back a Tensor
        return [get_item(index) for index in indices]

    # TODO: use keep dims and broadcast shape
    # note: this is a destructive operation
    def sum(self, axes: list[int], keepdim = False):
        if axes is None or axes == []:
            raise TypeError(f"Axes Cant be Null")
        if not isinstance(axes, list):
            axes = [axes]

        self.tensorData._init(self.ops.sum(self, axes))

    def transpose(self, axis1=0, axis2=1):
        self.tensorData.shape()[axis1], self.tensorData.shape()[axis2] = self.tensorData.shape()[axis2], self.tensorData.shape()[axis1]
        self.tensorData.stride()[axis1], self.tensorData.stride()[axis2] = self.tensorData.stride()[axis2], self.tensorData.stride()[axis1]

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
            res = self._init(tensorData)
            res.grad_fn = _backward
            res.parents = [self.tensorData, other.tensorData]
            return res
        elif isinstance(other, (int, float)):
            (tensorData, _backward) = self.ops.scalar_add(self, other)
            res = self._init(tensorData)
            res.grad_fn = _backward
            res.parents = [self.tensorData]
            return res
        raise TypeError(f"Can't add Tensor of type {self.dtype} with {type(other)}")

    def __sub__(self, other):
        if isinstance(other, Tensor):
            (tensorData, _backward) = self.ops.sub(self, other)
            res = self._init(tensorData)
            res.parents = [self.tensorData, other.tensorData]
            res.grad_fn = _backward
            return res
        elif isinstance(other, (int, float)):
            (tensorData, _backward) = self.ops.scalar_sub(self, other)
            res = self._init(tensorData)
            res.parents = [self.tensorData]
            res.grad_fn = _backward
            return res
        raise TypeError(f"Can't subtract Tensor of type {self.dtype} with {type(other)}")

    def __mul__(self, other):
        if isinstance(other, Tensor):
            (tensorData, _backward) = self.ops.mul(self, other)
            res = self._init(tensorData)
            res.grad_fn = _backward
            res.parents = [self.tensorData, other.tensorData]
            return res
        elif isinstance(other, (int, float)):
            (tensorData, _backward) = self.ops.scalar_mul(self, other)
            res = self._init(tensorData)
            res.grad_fn = _backward
            res.parents = [self.tensorData]
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
            return self._init(self.ops.matmul(self, other))
        raise TypeError(f"Can't divide Tensor of type {self.dtype} with {type(other)}")

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return self._init(self.ops.exp(self, other))
        if isinstance(other, (int, float)):
            return self._init(self.ops.scalar_exp(self, other))
        raise TypeError(f"Can't exponentiate Tensor of type {self.dtype} with {type(other)}")

    def __str__(self):
        shape = ', '.join(str(x) for x in self.tensorData.shape())
        return f"<{self.__class__.__module__}.{self.__class__.__name__}> (size: [{shape}], dtype={self.dtype})"

