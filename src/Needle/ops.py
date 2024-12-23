# later utilities for complete AD library
LAZY_MODE = False
TENSOR_COUNTER = 0

class ops:
    @staticmethod
    def add(tensor1, tensor2):
        return tensor1 + tensor2

    @staticmethod
    def sub(tensor1, tensor2):
        return tensor1 - tensor2

    @staticmethod
    def mul(tensor1, tensor2):
        return tensor1 * tensor2

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

    def matmul(self, data1, data2):
        return self.operations.mat_mul(data1, data2)
