from typing import List

from .tensor import Tensor
from .device import DeviceManager, TensorDtypes, TensorDevices

def randn(shape: List[int], mean = 0, std = 1, dtype="float32", device="cpu", requires_grad=False, debug_name=None) -> Tensor:
    dtype_enum = TensorDtypes(dtype)
    device_enum = TensorDevices(device)
    backend_tensor = DeviceManager.get_tensor(dtype_enum, device_enum)
    backend_ops = DeviceManager.get_backend(dtype_enum, device_enum)
    raw_tensor = backend_tensor.randn(shape, mean, std)
    return Tensor.create(raw_tensor, device, dtype, backend_ops, requires_grad, debug_name=debug_name)

def ones(shape: List[int], dtype="float32", device="cpu", requires_grad = False, debug_name=None) -> Tensor:
    dtype_enum = TensorDtypes(dtype)
    device_enum = TensorDevices(device)
    backend_tensor = DeviceManager.get_tensor(dtype_enum, device_enum)
    backend_ops = DeviceManager.get_backend(dtype_enum, device_enum)
    raw_tensor = backend_tensor.fill(shape, 1)
    return Tensor.create(raw_tensor, device, dtype, backend_ops, requires_grad, debug_name=debug_name)

def zeros(shape: List[int], dtype="float32", device="cpu", requires_grad=False, debug_name=None) -> Tensor:
    dtype_enum = TensorDtypes(dtype)
    device_enum = TensorDevices(device)
    backend_tensor = DeviceManager.get_tensor(dtype_enum, device_enum)
    backend_ops = DeviceManager.get_backend(dtype_enum, device_enum)
    raw_tensor = backend_tensor.fill(shape, 0)
    return Tensor.create(raw_tensor, device, dtype, backend_ops, requires_grad, debug_name=debug_name)