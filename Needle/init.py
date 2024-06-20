import random
from Needle.main import Needle

# initialization is being done in python not C++
def rand(mean = 0, std = 1, dtype = "float32"):
    if(dtype == "float32" or dtype == "float64"):
        return random.uniform(mean - std, mean + std)
    elif(dtype == "int32" or dtype == "int64"):
        return random.randint(mean - std, mean + std)
    else:
        raise ValueError("dtype %s is not supported" % dtype) 

def generate_ndarray(shape, mean, std, dtype):
    if len(shape) == 1:
        return [rand(mean, std, dtype) for _ in range(shape[0])]
    return [generate_ndarray(shape[1:], mean, std, dtype) for _ in range(shape[0])]

def randn(shape, mean = 0, std = 1, dtype="float32"):
    return Needle.Tensor(data = generate_ndarray(shape, mean, std, dtype), shape = shape, dtype = dtype)

def generate_n(shape, dtype, n):
    if len(shape) == 1:
        return [n] * shape[0]
    return [generate_n(shape[1:], dtype, n) for _ in range(shape[0])]

def ones(shape, dtype="float32"):
    return Needle.Tensor(data = generate_n(shape, dtype, 1), shape=shape, dtype=dtype)

def zeros(shape, dtype="float32"):
    return Needle.Tensor(data = generate_n(shape, dtype, 0), shape=shape, dtype=dtype)

# bind to Needle class
Needle.rand = staticmethod(rand)
Needle.randn = staticmethod(randn)
Needle.ones = staticmethod(ones)
Needle.zeros = staticmethod(zeros)
