import cpu_backend
import numpy as np

# later utilities for AD library
LAZY_MODE = False
TENSOR_COUNTER = 0

class  Needle():
    def __init__():
        pass

    # missing dtype
    class Tensor():
        def __init__(self, data, device="cpu"):
            self.device = "mps" if (device == "mps") else "cpu"
            # get our shape (currently implemented in python)
            self.shape = self.get_shape(self, data)
            self.data = data
            self._data = self.create_data_struct(self, data)

        @staticmethod
        def get_shape(self, array):
            if(self.device == "cpu"):
                if(isinstance(array, list)): # ND array
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
                self._data = [item for sublist in array for item in sublist]
                return cpu_backend.CPUBackend(self._data, self.shape)
            else:
                raise ValueError("Only CPU is currently implemented") 

    

x = Needle.Tensor([[1, 2, 3], [4, 5, 6]])
print(x.data)
print(x.shape)