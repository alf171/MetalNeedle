import sys; sys.path.append("tmp")
import backend

class DeviceManager:
    @staticmethod
    def set_dtype_tensor(dtype, device):
        # set backend device
        if device not in ["cpu", "mps"]:
            raise ValueError(f"device {device} is not supported")
        curBackend = backend.cpu if device == "cpu" else backend.gpu

        # put each behind a lambda for lazily evaluation
        backends = {
            "int32": lambda: (curBackend.IntTensor, curBackend.IntOperation()),
            "int64": lambda: (curBackend.LongTensor, curBackend.LongOperation),
            "float32": lambda: (curBackend.FloatTensor, curBackend.FloatOperation()),
            "float64": lambda: (curBackend.DoubleTensor, curBackend.DoubleOperation())
        }

        if dtype not in backends:
            raise ValueError(f"Data type {dtype} is not supported")
        return backends[dtype]()

