import cpu_backend

# later utilities for complete AD library
LAZY_MODE = False
TENSOR_COUNTER = 0

class  Needle():
    class Tensor():
        def __init__(self, data, device="cpu", dtype="int32"):
            self.device = "mps" if (device == "mps") else "cpu"
            (self._tensor, self._operations) = self.set_dtype_tensor(dtype)
            # get our shape (currently implemented in python)
            self.shape = self.get_shape(self, data)
            self.data = data
            self._data = self.create_data_struct(self, data)
            self.dtype = dtype

        @staticmethod
        def get_shape(self, array):
            if(self.device == "cpu"):
                if(isinstance(array, list)):
                    shape = []
                    current_level = array
                    while(isinstance(current_level, list)):
                        shape.append(len(current_level))
                        current_level = current_level[0] if len(current_level) > 0 else [] 
                    return shape
                else:
                    raise ValueError("Tensor data must be a list") 
            else:
                raise ValueError("Only CPU is currently implemented") 

        @staticmethod
        def create_data_struct(self, array):
            if(self.device == "cpu"):
                flat_data = [item for sublist in array for item in sublist]
                return self._tensor.initialize(flat_data, self.shape)
            else:
                raise ValueError("Only CPU is currently implemented") 
        
        @staticmethod
        def set_dtype_tensor(dtype):
            if dtype == "int32":
                return (cpu_backend.IntTensor, cpu_backend.CPUBackendInt())
            elif dtype == "int64":
                return (cpu_backend.LongTensor, cpu_backend.CPUBackendLong())
            elif dtype == "float32":
                return (cpu_backend.FloatTensor, cpu_backend.CPUBackendFloat())
            elif dtype == "float64":
                return (cpu_backend.DoubleTensor, cpu_backend.CPUBackendDouble())
            else:
                raise ValueError("dtype %s is not supported" % dtype) 

        def get_item(self, multi_dim_index):
            flat_index = self._data.mult_dim_to_flat_index(multi_dim_index)
            return self._data.data[flat_index]

        def __add__(self, other):
            result = Needle.Tensor.__new__(Needle.Tensor)
            result.device = self.device
            result.shape = self.shape
            # TODO: handle setting of data (only need this for prints really)
            result.data = None
            result._data = self._operations.ewise_add(self._data, other._data)
            return result
        
        def __mul__(self, other):
            result = Needle.Tensor.__new__(Needle.Tensor)
            result.device = self.device
            result.shape = self.shape
            # TODO: handle setting of data (only need this for prints really)
            result.data = None
            result._data = self._operations.ewise_mul(self._data, other._data)
            return result


x = Needle.Tensor([[1, 2, 3], [4, 5, 6]])
y = Needle.Tensor([[1, 2, 3], [4, 5, 6]])
z = x * y
print(z._data.data)