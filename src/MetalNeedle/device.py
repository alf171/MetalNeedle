import sys; sys.path.append("tmp")
import re
import backend

class DeviceManager:
    @staticmethod
    def set_dtype_tensor(dtype: str, device: str):
        # Validate device
        if device not in ["cpu", "metal"]:
            raise ValueError(f"device {device} is not supported")

        # Dynamically fetch backend attribute
        curBackend = getattr(backend, device, None)
        if curBackend is None:
            raise AttributeError(f"backend does not have attribute {device}")

        # Define backends with lazy evaluation
        backends = {
            "int32": lambda: (curBackend.IntTensor(), curBackend.IntOperation()),
            "int64": lambda: (curBackend.LongTensor(), curBackend.LongOperation()),
            "float32": lambda: (curBackend.FloatTensor(), curBackend.FloatOperation()),
            "float64": lambda: (curBackend.DoubleTensor(), curBackend.DoubleOperation()),
        }

        if dtype not in backends:
            raise ValueError(f"dtype {dtype} is not supported")

        return backends[dtype]()

    @staticmethod
    def is_tensor(obj):
        obj_type_str = str(type(obj))
        # TODO: could have a slightly more robust tensor check :)
        pattern = r"<class 'backend\.cpu\..*'>"
        return bool(re.match(pattern, obj_type_str))
