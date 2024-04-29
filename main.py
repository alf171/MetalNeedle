import numpy as np

# possibly not best class struct 
# I might decentralize everything
class  Needle():
    def __init__():
        pass

    class Tensor():
        def __init__(self, array, device="cpu"):
            self.device = "mps" if (device == "mps") else "cpu"
            self.array = self.create_data_struct(array, self.device)

        # for now we are a python list so this is fine
        def __str__(self):
            return str(self.array)

        # if i want a better debug print - maybe I'll use numpy for this
        def __repr__(self):
            pass

        @staticmethod
        def create_data_struct(array, device):
            # need some routing logic here based on device
            return array

    class Op():

        def scalar_add(self, a, scalar):
            if not isinstance(a, Needle.Tensor):
                raise ValueError("Argument is not a supported tensor")
            return a + scalar
        


x = Needle.Tensor([1,2,3])
print(x)