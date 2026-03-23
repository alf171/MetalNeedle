import unittest

from MetalNeedle.data import TensorData


class TestDivisionViews(unittest.TestCase):
    def test_division_works_with_non_contiguous_broadcast_view(self):
        numerator = TensorData([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]], "float32", "cpu")
        denominator = TensorData([[2.0, 2.0, 3.0]], "float32", "cpu").broadcast([2, 3])

        self.assertFalse(denominator.raw_tensor.is_contiguous())

        result = numerator / denominator

        self.assertEqual(result.shape(), [2, 3])
        self.assertAlmostEqual(result.get_single_item([0, 0]), 1.0)
        self.assertAlmostEqual(result.get_single_item([0, 1]), 2.0)
        self.assertAlmostEqual(result.get_single_item([0, 2]), 2.0)
        self.assertAlmostEqual(result.get_single_item([1, 0]), 4.0)
        self.assertAlmostEqual(result.get_single_item([1, 1]), 5.0)
        self.assertAlmostEqual(result.get_single_item([1, 2]), 4.0)


if __name__ == "__main__":
    unittest.main()
