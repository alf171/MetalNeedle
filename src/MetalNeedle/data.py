from .util import ShapeUtils


class TensorData:
    def __init__(self, data, _tensor):
        _shape = ShapeUtils.get_shape(data)
        self.tensor = ShapeUtils.create_data_struct(_tensor, data, _shape)

    @staticmethod
    def create(data):
        result = TensorData.__new__(TensorData)
        result.tensor = data
        return result

    def _init(self, tensor):
        self.tensor = tensor

    def shape(self):
        return self.tensor.shape

    def setShape(self, shape):
        self.tensor.shape = shape

    def stride(self):
        return self.tensor.stride

    def setStride(self, shape):
        self.tensor.stride = shape

    def __getitem__(self, index):
        return self.tensor.data[index]

    def mult_dim_to_flat_index(self, idx):
        return self.tensor.mult_dim_to_flat_index(idx)
