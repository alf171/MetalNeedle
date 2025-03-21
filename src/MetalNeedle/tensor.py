from .util import ShapeUtils
from .ops import TensorOperations
from .device import DeviceManager
from .data import TensorData

class Tensor:
    def __init__(self, data, device="cpu", dtype="int32", requires_grad=False, debug_name=None):
        self.device: str = device
        self.dtype: str = dtype
        self.tensor_data: TensorData = TensorData(data, dtype, device, debug_name)
        self.ops = TensorOperations
        # autograd related
        self.parents: list[Tensor] = []
        self.requires_grad: bool = requires_grad
        self.grad: TensorData or None = None
        self.grad_fn = None

    def clone(self):
        """
        Creates a deep copy of the tensor with completely independent memory.

        Returns:
            Tensor: A new tensor with identical values but separate memory
        """
        # _, backend_ops = DeviceManager.set_dtype_tensor(self.dtype, self.device)
        return Tensor.create(
            raw_tensor=self.data,
            device=self.device,
            dtype=self.dtype,
            operations=self.tensor_data.operations,
            requires_grad=self.requires_grad,
            debug_name=f"{self._debug_name()}_clone" if self._debug_name else "cloned_tensor"
        )

    @staticmethod
    def create(raw_tensor, device: str, dtype: str, operations, requires_grad=False, debug_name=None):
        result = Tensor.__new__(Tensor)
        result.device = device
        result.dtype = dtype
        result.ops = TensorOperations
        result.tensor_data = TensorData.create(raw_tensor, operations, debug_name)
        result.requires_grad = requires_grad
        result.grad = None
        result.grad_fn = None
        return result

    def _init(self, data, _grad_fn = None, debug_name = None):
        result = Tensor.__new__(Tensor)
        result.device = self.device
        result.dtype = self.dtype
        result.ops = self.ops
        # result._operations = self._operations
        # result._tensor = self._tensor
        result.tensor_data = data
        result.tensor_data._debug_name = debug_name
        result.requires_grad = self.requires_grad
        result.grad_fn = _grad_fn
        result.grad = None
        return result

    def shape(self):
        return self.tensor_data.shape()

    def data(self):
        return self.tensor_data.data()

    def _debug_name(self):
        return self.tensor_data._debug_name

    # TODO: support partial slicing and return a tensor if sum(size) > 1
    def __getitem__(self, multi_dim_index):
        if not isinstance(multi_dim_index, (list, tuple)):
            multi_dim_index = [multi_dim_index]

        if len(multi_dim_index) > len(self.tensor_data.shape()):
            raise ValueError("index shape exceeds tensor's dimensions")

        res = []
        for (index, dim) in zip(multi_dim_index, self.tensor_data.shape()):
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
            flat_index = self.tensor_data.mult_dim_to_flat_index(idx)
            return self.tensor_data[flat_index]

        if len(indices) == 1:
            return get_item(indices[0])

        # could also consider returning back a Tensor if sum(tensor.size()) > 1
        return [get_item(index) for index in indices]

    def transpose(self, debug_name = None):
        (tensor_data, _grad_fn) = self.ops.swap(self, 0, 1)
        res = self._init(tensor_data, _grad_fn, debug_name)
        res.parents = [self]
        return res

    def swap(self, axis1, axis2):
        # if axis is negative, index opposite direction
        # consider moving this code lower down the stack
        if axis2 < 0:
            axis2 = len(self.shape()) + axis2
        self.ops.swap(self, axis1, axis2)

    def reshape(self, new_shape: list[int], debug_name = None):
        (tensor_data, _grad_fn) = self.ops.reshape(self, new_shape)
        res = self._init(tensor_data, _grad_fn, debug_name)
        res.parents = [self]
        return res

    def __add__(self, other, debug_name=None):
        # do we not assert shape is same
        if isinstance(other, Tensor):
            (tensor_data, _grad_fn) = self.ops.add(self, other)
            res = self._init(tensor_data, _grad_fn, debug_name)
            res.parents = [self, other]
            return res
        elif isinstance(other, (int, float)):
            (tensor_data, _grad_fn) = self.ops.scalar_add(self, other)
            res = self._init(tensor_data, _grad_fn)
            res.parents = [self]
            return res
        raise TypeError(f"Can't add Tensor of type {self.dtype} with {type(other)}")

    def __sub__(self, other):
        if isinstance(other, Tensor):
            (tensor_data, _grad_fn) = self.ops.sub(self, other)
            res = self._init(tensor_data, _grad_fn)
            res.parents = [self, other]
            return res
        elif isinstance(other, (int, float)):
            (tensor_data, _grad_fn) = self.ops.scalar_sub(self, other)
            res = self._init(tensor_data, _grad_fn)
            res.parents = [self]
            return res
        raise TypeError(f"Can't subtract Tensor of type {self.dtype} with {type(other)}")

    def __mul__(self, other, debug_name=None):
        if isinstance(other, Tensor):
            (tensor_data, _grad_fn) = self.ops.mul(self, other)
            res = self._init(tensor_data, _grad_fn, debug_name)
            res.parents = [self, other]
            return res
        elif isinstance(other, (int, float)):
            (tensor_data, _grad_fn) = self.ops.scalar_mul(self, other)
            res = self._init(tensor_data, _grad_fn, debug_name)
            res.parents = [self]
            return res
        raise TypeError(f"Can't multiply Tensor of type {self.dtype} with {type(other)}")

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            (tensor_data, _grad_fn) = self.ops.div(self, other)
            res = self._init(tensor_data, _grad_fn)
            res.parents = [self, other]
            return res
        elif isinstance(other, (int, float)):
            (tensor_data, _grad_fn) = self.ops.scalar_div(self, other)
            res = self._init(tensor_data, _grad_fn)
            res.parents = [self]
            return res
        raise TypeError(f"Can't divide Tensor of type {self.dtype} with {type(other)}")

    def __pow__(self, other):
        if isinstance(other, Tensor):
            (tensor_data, _grad_fn) = self.ops.exp(self, other)
            res = self._init(tensor_data, _grad_fn)
            res.parents = [self, other]
            return res
        if isinstance(other, (int, float)):
            (tensor_data, _grad_fn) = self.ops.scalar_exp(self, other)
            res = self._init(tensor_data, _grad_fn)
            res.parents = [self]
            return res
        raise TypeError(f"Can't exponentiate Tensor of type {self.dtype} with {type(other)}")

    def log(self):
        (tensor_data, _grad_fn) = self.ops.scalar_log(self)
        res = self._init(tensor_data, _grad_fn)
        res.parents = [self]
        return res

    def __matmul__(self, other):
        if isinstance(other, Tensor):
            (tensor_data, _grad_fn) = self.ops.matmul(self, other)
            res = self._init(tensor_data, _grad_fn)
            res.parents = [self, other]
            return res

        raise TypeError(f"Can't divide Tensor of type {self.dtype} with {type(other)}")

    # TODO: use keep dims and broadcast shape
    # also, this shouldn't be destructive
    def sum(self, axes: int or list[int], keepdim = False):
        if axes is None or axes == []:
            raise TypeError(f"Axes Cant be Null")
        if not isinstance(axes, list):
            axes = [axes]

        (tensor_data, _grad_fn) = self.ops.sum(self, axes, keepdim)
        res = self._init(tensor_data, _grad_fn)
        res.parents = [self]
        return res

    def broadcast(self, new_shape: list[int]):
        tensor_data, _grad_fn = self.ops.broadcast(self, new_shape)
        res = self._init(tensor_data, _grad_fn)
        res.parents = [self]
        return res

    def backward(self, grad=None):
        if grad is None:
            grad = self.tensor_data.ones_like()

        # this is where we do += on the grad so it is not require on the operation
        self.grad = grad if self.grad is None else TensorData.__add__(self.grad, grad)
        if self._debug_name() is not None:
            self.grad._debug_name = self._debug_name() + "_grad"

        if self.grad_fn is not None:
            self.grad_fn(grad)

    def __str__(self):
        shape = ', '.join(str(x) for x in self.shape())
        return f"<{self.__class__.__module__}.{self.__class__.__name__}> (size: [{shape}], dtype={self.dtype})"
