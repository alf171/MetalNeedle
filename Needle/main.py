import sys; sys.path.insert(1, "tmp")
import cpu_backend
# import gpu_backend

# later utilities for complete AD library
LAZY_MODE = False
TENSOR_COUNTER = 0

class  Needle():
    # TODO: add automatic differentiation
    class Tensor():
        def __init__(self, data, shape = None, device="cpu", dtype="int32"):
            self.device = "mps" if (device == "mps") else "cpu"
            (self._tensor, self._operations) = self.set_dtype_tensor(dtype, self.device)
            # get our shape (currently implemented in python)
            self.shape = self.get_shape(self, data) if shape == None else shape

            self._data = self.create_data_struct(self, data)
            self.dtype = dtype

        @staticmethod
        def get_shape(self, array):
            if(self.device == "cpu" or self.device == "mps"):
                if(isinstance(array, list)):
                    shape = []
                    current_level = array
                    while(isinstance(current_level, list)):
                        shape.append(len(current_level))
                        current_level = current_level[0] if len(current_level) > 0 else [] 
                    return shape
                else:
                    raise ValueError("tensor data must be a list") 
            else:
                raise ValueError(f"device {self.device} is not supported") 

        @staticmethod
        def create_data_struct(self, array):
            flat_data = [item for sublist in array for item in sublist]
            return self._tensor.initialize(flat_data, self.shape)
            
        # TODO: fix pretty repetitive code
        @staticmethod
        def set_dtype_tensor(dtype, device):
            if device == "cpu":
                backend = cpu_backend
            elif device == "mps":
                backend = gpu_backend
            else:
                raise ValueError("device %s is not supported")
            
            if dtype == "int32":
                return (backend.IntTensor, backend.CPUBackendInt())
            elif dtype == "int64":
                return (backend.LongTensor, backend.CPUBackendLong())
            elif dtype == "float32":
                return (backend.FloatTensor, backend.CPUBackendFloat())
            elif dtype == "float64":
                return (backend.DoubleTensor, backend.CPUBackendDouble())
            else:
                raise ValueError("dtype %s is not supported" % dtype) 

        def get_item(self, multi_dim_index):
            flat_index = self._data.mult_dim_to_flat_index(multi_dim_index)
            return self._data.data[flat_index]

        def __add__(self, other):
            result = Needle.Tensor.__new__(Needle.Tensor)
            result.device = self.device
            result.shape = self.shape
            result.dtype = self.dtype
            result._data = self._operations.ewise_add(self._data, other._data)
            return result
        
        def __mul__(self, other):
            result = Needle.Tensor.__new__(Needle.Tensor)
            result.device = self.device
            result.shape = self.shape
            result.dtype = self.dtype
            result._data = self._operations.ewise_mul(self._data, other._data)
            return result
        
        def __matmul__(self, other):
            result = Needle.Tensor.__new__(Needle.Tensor)
            result.device = self.device
            result.shape = self.shape[:-1] + other.shape[1:]
            result.dtype = self.dtype
            result._data = self._operations.mat_mul(self._data, other._data)
            return result 

        # print shape and dtype in addition to default stuff
        def __str__(self):
            shape = ', '.join(str(x) for x in self.shape)
            return f"<{self.__class__.__module__}.{self.__class__.__name__} object at {hex(id(self))}> (size: [{shape}], dtype={self.dtype})"

