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
        cur_backend = getattr(backend, device, None)
        if cur_backend is None:
            raise AttributeError(f"backend does not have attribute {device}")

        # Define backends with lazy evaluation
        backends = {
            "int32": lambda: (cur_backend.IntTensor(), cur_backend.IntOperation()),
            "int64": lambda: (cur_backend.LongTensor(), cur_backend.LongOperation()),
            "float32": lambda: (cur_backend.FloatTensor(), cur_backend.FloatOperation()),
            "float64": lambda: (cur_backend.DoubleTensor(), cur_backend.DoubleOperation()),
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
