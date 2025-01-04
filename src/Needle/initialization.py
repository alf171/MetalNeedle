from .tensor import Tensor
from .device import DeviceManager

def randn(shape, mean = 0, std = 1, dtype="float32", device="cpu"):
    backendTensor, backendOps = DeviceManager.set_dtype_tensor(dtype, device)
    rawData = backendTensor().randn(shape, mean, std)
    _data = backendTensor().initialize(rawData, shape)
    return Tensor._init(None, data=_data, device=device, shape=shape, dtype=dtype, ops=backendOps)

def ones(shape, dtype="float32", device="cpu"):
    backendTensor, backendOps = DeviceManager.set_dtype_tensor(dtype, device)
    rawData = backendTensor().create(shape, 1)
    _data = backendTensor().initialize(rawData, shape)
    return Tensor._init(None, data=_data, device=device, shape=shape, dtype=dtype, ops=backendOps)

# TODO: if slow, more over to C++
def zeros(shape, dtype="float32", device="cpu"):
    backendTensor, backendOps = DeviceManager.set_dtype_tensor(dtype, device)
    rawData = backendTensor().create(shape, 0)
    _data = backendTensor().initialize(rawData, shape)
    return Tensor._init(None, data=_data, device=device, shape=shape, dtype=dtype, ops=backendOps)
