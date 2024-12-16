from .init import *
# later utilities for complete AD library
LAZY_MODE = False
TENSOR_COUNTER = 0

class ops():
    class add():
        def compute(self, tensor1: Needle.Tensor, tensor2: Needle.Tensor):
            return tensor1 + tensor2

        def gradient(self):
            return

    def sub(tensor1: Needle.Tensor, tensor2: Needle.Tensor):
        return tensor1 + tensor2

    def mul(tensor1: Needle.Tensor, tensor2: Needle.Tensor):
        return tensor1 * tensor2

    def matmul(tensor1: Needle.Tensor, tensor2: Needle.Tensor):
        return tensor1 @ tensor2


