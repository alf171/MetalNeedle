from main import Needle

class Op():
    def scalar_add(self, a, scalar):
        if not isinstance(a, Needle.Tensor):
            raise ValueError("Argument is not a supported tensor")
        