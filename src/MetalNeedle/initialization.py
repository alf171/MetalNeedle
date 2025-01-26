from .tensor import Tensor
from .device import DeviceManager

def randn(shape: list[int], mean = 0, std = 1, dtype="float32", device="cpu", requires_grad=False) -> Tensor:
    backendTensor, backendOps = DeviceManager.set_dtype_tensor(dtype, device)
    rawData = backendTensor.randn(shape, mean, std)
    _data = backendTensor.initialize(rawData, shape)
    return Tensor.create(_data, device, dtype, backendTensor, backendOps, requires_grad)

def ones(shape: list[int], dtype="float32", device="cpu", requires_grad = False) -> Tensor:
    backendTensor, backendOps = DeviceManager.set_dtype_tensor(dtype, device)
    rawData = backendTensor.create(shape, 1)
    _data = backendTensor.initialize(rawData, shape)
    return Tensor.create(_data, device, dtype, backendTensor, backendOps, requires_grad)

def zeros(shape: list[int], dtype="float32", device="cpu", requires_grad=False) -> Tensor:
    backendTensor, backendOps = DeviceManager.set_dtype_tensor(dtype, device)
    rawData = backendTensor.create(shape, 0)
    _data = backendTensor.initialize(rawData, shape)
    return Tensor.create(_data, device, dtype, backendTensor, backendOps, requires_grad)