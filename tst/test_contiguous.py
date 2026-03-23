import unittest

from MetalNeedle.data import TensorData


class TestContiguous(unittest.TestCase):
    def test_fresh_tensor_is_contiguous(self):
        tensor = TensorData([[1.0, 2.0], [3.0, 4.0]], "float32", "cpu")
        self.assertTrue(tensor.raw_tensor.is_contiguous())

    def test_transpose_view_is_not_contiguous_until_compacted(self):
        tensor = TensorData([[1.0, 2.0], [3.0, 4.0]], "float32", "cpu")
        transposed = tensor.transpose()

        self.assertFalse(transposed.raw_tensor.is_contiguous())
        transposed.compact()
        self.assertTrue(transposed.raw_tensor.is_contiguous())

    def test_broadcast_view_is_not_contiguous(self):
        tensor = TensorData([[1.0, 2.0, 3.0]], "float32", "cpu")
        broadcasted = tensor.broadcast([4, 3])

        self.assertFalse(broadcasted.raw_tensor.is_contiguous())
        self.assertTrue(tensor.raw_tensor.is_contiguous())


if __name__ == "__main__":
    unittest.main()
