import sys;
from enum import Enum

sys.path.append("tmp")
import re
import backend


class TensorDtypes(Enum):
    byte = "int8"
    int = "int32"
    long = "int64"
    float = "float32"
    double = "float64"

DTYPE_TO_SIZE_ENCODE = {
    TensorDtypes.byte: 1,
    TensorDtypes.int: 4,
    TensorDtypes.long: 8,
    TensorDtypes.float: 4,
    TensorDtypes.double: 8,
}

DTYPE_TO_ARRAY_ENCODE = {
    TensorDtypes.byte: 'b',
    TensorDtypes.int: 'i',
    TensorDtypes.long: 'q',
    TensorDtypes.float: 'f',
    TensorDtypes.double: 'd'
}

class TensorDevices(Enum):
    cpu = "cpu"
    metal = "metal"

    def __str__(self):
        return self.value


class DeviceManager:
    @staticmethod
    def get_tensor(dtype: TensorDtypes, device: TensorDevices):
        cur_backend = getattr(backend, device.__str__(), None)
        if cur_backend is None:
            raise AttributeError(f"backend does not have attribute {device}")

        tensors = {
            TensorDtypes.byte: lambda: cur_backend.ByteTensor(),
            TensorDtypes.int: lambda: cur_backend.IntTensor(),
            TensorDtypes.long: lambda: cur_backend.LongTensor(),
            TensorDtypes.float: lambda: cur_backend.FloatTensor(),
            TensorDtypes.double: lambda: cur_backend.DoubleTensor(),
        }

        if dtype not in tensors:
            raise ValueError(f"dtype {dtype} is not supported")

        return tensors[dtype]()

    @staticmethod
    def get_backend(dtype: TensorDtypes, device: TensorDevices):
        cur_backend = getattr(backend, device.__str__(), None)
        if cur_backend is None:
            raise AttributeError(f"backend does not have attribute {device}")

        # Define backends with lazy evaluation
        backends = {
            TensorDtypes.byte: lambda: cur_backend.ByteOperation(),
            TensorDtypes.int: lambda: cur_backend.IntOperation(),
            TensorDtypes.long: lambda: cur_backend.LongOperation(),
            TensorDtypes.float: lambda: cur_backend.FloatOperation(),
            TensorDtypes.double: lambda: cur_backend.DoubleOperation(),
        }

        if dtype not in backends:
            raise ValueError(f"dtype {dtype} is not supported")

        return backends[dtype]()

    @staticmethod
    def is_tensor(obj):
        obj_type_str = str(type(obj))
        # TODO: could have a slightly more robust tensor check :)
        pattern = r"<class 'backend\.(cpu|metal)\..*'>"
        return bool(re.match(pattern, obj_type_str))
