from abc import ABC, abstractmethod

"""Abstract base class for tensor backends."""
class TensorBackend(ABC):
    @abstractmethod
    def initialize_tensor(self, data, shape):
        pass

    @abstractmethod
    def ewise_add(self, tensor1, tensor2):
        pass

    @abstractmethod
    def ewise_mul(self, tensor1, tensor2):
        pass

    @abstractmethod
    def mat_mul(self, tensor1, tensor2):
        pass

    @abstractmethod
    def  testing(self, tensor1):
        pass