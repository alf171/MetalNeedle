import unittest

from MetalNeedle.data import TensorData


class TestBroadcast(unittest.TestCase):
    def test_broadcast_expands_with_zero_stride_view(self):
        source = TensorData([[1.0, 2.0, 3.0]], "float32", "cpu")
        broadcasted = source.broadcast([3, 3])

        self.assertEqual(source.shape(), [1, 3])
        self.assertEqual(broadcasted.shape(), [3, 3])

        for row in range(3):
            self.assertEqual(broadcasted.get_single_item([row, 0]), 1.0)
            self.assertEqual(broadcasted.get_single_item([row, 1]), 2.0)
            self.assertEqual(broadcasted.get_single_item([row, 2]), 3.0)

    def test_broadcast_shares_storage_without_mutating_source_metadata(self):
        source = TensorData([[1.0, 2.0, 3.0]], "float32", "cpu")
        broadcasted = source.broadcast([3, 3])

        broadcasted[[2, 0]] = 99.0

        self.assertEqual(source.shape(), [1, 3])
        self.assertEqual(broadcasted.shape(), [3, 3])
        self.assertEqual(source.get_single_item([0, 0]), 99.0)
        self.assertEqual(broadcasted.get_single_item([0, 0]), 99.0)
        self.assertEqual(broadcasted.get_single_item([2, 0]), 99.0)


if __name__ == "__main__":
    unittest.main()
