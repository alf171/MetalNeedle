import unittest

from MetalNeedle.data import TensorData


class TestView(unittest.TestCase):
    def test_view_shares_underlying_storage(self):
        source = TensorData([[1.0, 2.0], [3.0, 4.0]], "float32", "cpu")
        view = source.view("alias")

        view[[0, 0]] = 99.0

        self.assertEqual(source.get_single_item([0, 0]), 99.0)
        self.assertEqual(view.get_single_item([0, 0]), 99.0)
        self.assertEqual(source.shape(), [2, 2])
        self.assertEqual(view.shape(), [2, 2])

    def test_view_can_change_metadata_without_changing_source_shape(self):
        source = TensorData([[1.0, 2.0], [3.0, 4.0]], "float32", "cpu")
        view = source.view()

        view.set_shape([4])
        view.set_stride([1])

        self.assertEqual(source.shape(), [2, 2])
        self.assertEqual(view.shape(), [4])
        self.assertEqual(source.data(), view.data())


if __name__ == "__main__":
    unittest.main()
