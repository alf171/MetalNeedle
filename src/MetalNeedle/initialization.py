from .tensor import Tensor
from .device import DeviceManager

def randn(shape: list[int], mean = 0, std = 1, dtype="float32", device="cpu", requires_grad=False, debug_name=None) -> Tensor:
    backend_tensor, backend_ops = DeviceManager.set_dtype_tensor(dtype, device)
    raw_data = backend_tensor.randn(shape, mean, std)
    _data = backend_tensor.initialize(raw_data, shape)
    return Tensor.create(_data, device, dtype, backend_ops, requires_grad, debug_name=debug_name)

def ones(shape: list[int], dtype="float32", device="cpu", requires_grad = False, debug_name=None) -> Tensor:
    backend_tensor, backend_ops = DeviceManager.set_dtype_tensor(dtype, device)
    raw_data = backend_tensor.create(shape, 1)
    raw_tensor = backend_tensor.initialize(raw_data, shape)
    return Tensor.create(raw_tensor, device, dtype, backend_ops, requires_grad, debug_name=debug_name)

def zeros(shape: list[int], dtype="float32", device="cpu", requires_grad=False, debug_name=None) -> Tensor:
    backend_tensor, backend_ops = DeviceManager.set_dtype_tensor(dtype, device)
    raw_data = backend_tensor.create(shape, 0)
    raw_tensor = backend_tensor.initialize(raw_data, shape)
    return Tensor.create(raw_tensor, device, dtype, backend_ops, requires_grad, debug_name=debug_name)