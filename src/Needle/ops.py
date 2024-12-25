# later utilities for complete AD library
LAZY_MODE = False
TENSOR_COUNTER = 0

# for gradient L res[0] will represent gradient w.r.t tensor1
# and R res[1] will represent grad of tensor2
# TODO: add automatic differentiation
class ops:
    class add:
        def compute(self, tensor1, tensor2):
            return tensor1 + tensor2

        def gradient(self, tensor1, tensor2):
            return [1, 1]

    class sub:
        def compute(self, tensor1, tensor2):
            return tensor1 - tensor2

        def gradient(self, tensor1, tensor2):
            return [1, -1]

    class mul:
        def compute(self, tensor1, tensor2):
            return tensor1 * tensor2

        def gradient(self, tensor1, tensor2):
            return [tensor2, tensor1]

    class div:
        def compute(self, tensor1, tensor2):
            return tensor1 / tensor2

        def gradient(self, tensor1, tensor2):
            return [1 / tensor2, ((-1 * tensor1) / (tensor2**2))]

    class matmul:
        def compute(self, tensor1, tensor2):
            return tensor1 @ tensor2

        def gradient(self, tensor1, tensor2):
            return [tensor2, tensor1]

    @staticmethod
    def matmul(tensor1, tensor2):
        return tensor1 @ tensor2

class TensorOperations:
    def __init__(self, operations):
        self.operations = operations

    def add(self, data1, data2):
        return self.operations.ewise_add(data1, data2)

    def scalar_add(self, data1, value):
        return self.operations.scalar_add(data1, value)

    def sub(self, data1, data2):
        return self.operations.ewise_add(data1, data2)

    def scalar_sub(self, data1, value):
        return self.operations.scalar_add(data1, value)

    def mul(self, data1, data2):
        return self.operations.ewise_mul(data1, data2)

    def scalar_mul(self, data1, value):
        return self.operations.scalar_mul(data1, value)

    def div(self, data1, data2):
        return self.operations.ewise_div(data1, data2)

    def scalar_div(self, data1, value):
        return self.operations.scalar_div(data1, value)

    def div(self, data1, data2):
        return self.operations.ewise_exp(data1, data2)

    def scalar_div(self, data1, value):
        return self.operations.scalar_exp(data1, value)

    def matmul(self, data1, data2):
        return self.operations.mat_mul(data1, data2)
