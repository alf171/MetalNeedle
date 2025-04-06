from .tensor import Tensor
from .device import DeviceManager

def randn(shape: list[int], mean = 0, std = 1, dtype="float32", device="cpu", requires_grad=False, debug_name=None) -> Tensor:
    backend_tensor, backend_ops = DeviceManager.set_dtype_tensor(dtype, device)
    raw_tensor = backend_tensor.randn(shape, mean, std)
    return Tensor.create(raw_tensor, device, dtype, backend_ops, requires_grad, debug_name=debug_name)

def ones(shape: list[int], dtype="float32", device="cpu", requires_grad = False, debug_name=None) -> Tensor:
    backend_tensor, backend_ops = DeviceManager.set_dtype_tensor(dtype, device)
    raw_tensor = backend_tensor.fill(shape, 1)
    return Tensor.create(raw_tensor, device, dtype, backend_ops, requires_grad, debug_name=debug_name)

def zeros(shape: list[int], dtype="float32", device="cpu", requires_grad=False, debug_name=None) -> Tensor:
    backend_tensor, backend_ops = DeviceManager.set_dtype_tensor(dtype, device)
    raw_tensor = backend_tensor.fill(shape, 0)
    return Tensor.create(raw_tensor, device, dtype, backend_ops, requires_grad, debug_name=debug_name)