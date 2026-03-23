import math
import os
import tempfile
import MetalNeedle as mn
import unittest

class TestTensor(unittest.TestCase):
    def test_ThreeByThreeMatMulCheck(self):
        x = mn.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = mn.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        z = x @ y

        expected_result = [
            [30, 36, 42],
            [66, 81, 96],
            [102, 126, 150],
        ]

        self.assertEqual(z.shape(), [3,3])

        for i in range(3):
            for j in range(3):
                expected_value = expected_result[i][j]
                calculated_value = z[i, j]
                self.assertEqual(expected_value, calculated_value)


    def test_ScalarOperations(self):
        x = mn.ones([3, 3])
        z = (x + 3)
        self.assertEqual(z[1,1], 4)
        z = (z - 2)
        self.assertEqual(z[1,1], 2)
        z = (z * 3)
        self.assertEqual(z[1,1], 6)
        z = (z ** 2)
        self.assertEqual(z[1,1], 36)
        z = (z / 2)
        self.assertEqual(z[1,1], 18)

        z = z.log()
        self.assertAlmostEqual(z[1,1], math.log(18))

    def test_SlicingOperations(self):
        x1 = mn.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        y1 = (x1[1,:])
        self.assertEqual(y1.shape(), [1,3])
        self.assertEqual(y1[0,0], 4)
        self.assertEqual(y1[0,1], 5)
        self.assertEqual(y1[0,2], 6)

        x2 = mn.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        y2 = (x2[:,1])
        self.assertEqual(y2.shape(), [3,1])
        self.assertEqual(y2[0,0], 2)
        self.assertEqual(y2[1,0], 5)
        self.assertEqual(y2[2,0], 8)

    def test_SumOperation(self):
        x1 = mn.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        x2 = x1.sum(0)
        self.assertEqual(x2[0], 12)
        self.assertEqual(x2[1], 15)
        self.assertEqual(x2[2], 18)
        self.assertEqual(x2.shape(), [3])

        x3 = mn.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        x4 = x3.sum(1)
        self.assertEqual(x4.shape(), [3])
        self.assertEqual(x4[0], 6)
        self.assertEqual(x4[1], 15)
        self.assertEqual(x4[2], 24)

        x5 = mn.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        x6 = x5.sum([0,1])
        self.assertEqual(x6[0], ((9*10)/2))
        self.assertEqual(x6.shape(), [1])

    def test_BroadcastOperation(self):
        x1 = mn.Tensor([1, 2, 3])
        x2 = x1.reshape([1,3])
        x3 = x2.broadcast([3,3])
        for i in range(3):
            self.assertEqual(x3[i,0], 1)
            self.assertEqual(x3[i,1], 2)
            self.assertEqual(x3[i,2], 3)

    def test_ReshapeOperations(self):
        x = mn.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        x = x.transpose()
        self.assertEqual(x[0,0], 1)
        self.assertEqual(x[0,1], 4)
        self.assertEqual(x[0,2], 7)
        self.assertEqual(x[2,2], 9)
        x = mn.Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        x = x.reshape([9, 1])
        self.assertEqual(x[0,0], 1)
        self.assertEqual(x[8,0], 3)
        x = x.transpose()
        self.assertEqual(x[0,8], 3)

    def test_Autograd(self):
        a = mn.Tensor([[1, 2], [3, 4]], requires_grad=True, debug_name="a")
        b = mn.Tensor([[5, 6], [7, 8]], requires_grad=True, debug_name="b")
        c = mn.Tensor([[9, 10], [11, 12]], requires_grad=True, debug_name="c")
        x1 = mn.Tensor.__add__(a, b, debug_name="x1")
        x2 = mn.Tensor.__mul__(x1, c, debug_name="x2")
        y1 = mn.Tensor.__mul__(a, c, debug_name="y1")
        z = mn.Tensor.__add__(x2, y1, debug_name="z")
        z.backward()
        self.assertEqual(a.grad.data(), [18, 20, 22, 24])
        self.assertEqual(b.grad.data(), [9, 10, 11, 12])
        self.assertEqual(c.grad.data(), [7, 10, 13, 16])

        a = mn.ones([5, 3], requires_grad=True, debug_name="a")
        b = mn.ones([3, 5], requires_grad=True, debug_name="a")
        res = a @ b
        res.backward()
        self.assertEqual(a.grad.shape(), [5,3])

    # def test_Pytorch():
    #     a = torch.ones([5,3], requires_grad=True)
    #     b = torch.ones([3,5], requires_grad=True)
    #     #
    #     # # Create computational graph
    #     z = a @ b
    #     #
    #     z.backward(torch.ones_like(z))
    #     #
    #     # # Print computed gradients
    #     # print("Computed gradients:")
    #     # print(a.grad)
    #     # print(f"b.grad = \n{b.grad}")  # Should be c
    #     # print(f"c.grad = \n{c.grad}")  # Should be 2a + b


    def test_MaximumsAndMinimum(self):
        a = mn.Tensor([[-1, 14], [3, 4]])
        self.assertEqual(a.max()[0], 14)

        b = mn.Tensor([[2, 13], [9, 3]])
        c = a.maximum(b)
        self.assertEqual(c[0,0], 2)
        self.assertEqual(c[0,1], 14)
        self.assertEqual(c[1,0], 9)
        self.assertEqual(c[1,1], 4)

        # a ([[-1, 14], [3, 4]])
        # b ([[2, 13], [9, 3]])
        d = a.minimum(b)
        self.assertEqual(d[0,0], -1)
        self.assertEqual(d[0,1], 13)
        self.assertEqual(d[1,0], 3)
        self.assertEqual(d[1,1], 3)

        e = a.maximum(0)
        self.assertEqual(e[0,0], 0)

        f = mn.Tensor([[1,1,1], [2,2,2]])
        self.assertEqual(f.max([0])[0], 2)
        self.assertEqual(f.max([0])[1], 2)
        self.assertEqual(f.max([0])[2], 2)
        self.assertEqual(f.max([1])[0], 1)
        self.assertEqual(f.max([1])[1], 2)

    def test_clip(self):
        x = mn.Tensor([[1, 2, 3], [4, 5, 6]], dtype="float32")

        # const, const
        result1 = x.clip(2, 5)
        expected1 = mn.Tensor([[2, 2, 3], [4, 5, 5]], dtype="float32")
        # self.assertTrue(np.allclose(result1.data(), expected1.data()))

        # const, tensor
        upper_tensor = mn.Tensor([[3, 5, 4], [5, 4, 3]], dtype="float32")
        result2 = x.clip(2, upper_tensor)
        expected2 = mn.Tensor([[2, 2, 3], [4, 4, 3]], dtype="float32")
        # self.assertTrue(np.allclose(result2.data(), expected2.data()))

        # tensor, const
        lower_tensor = mn.Tensor([[0, 3, 2], [3, 4, 7]], dtype="float32")
        result3 = x.clip(lower_tensor, 5)
        expected3 = mn.Tensor([[1, 3, 3], [4, 5, 5]], dtype="float32")
        # self.assertTrue(np.allclose(result3.data(), expected3.data()))

        # tensor, tensor
        result4 = x.clip(lower_tensor, upper_tensor)
        expected4 = mn.Tensor([[1, 3, 3], [4, 4, 6]], dtype="float32")
        # self.assertTrue(np.allclose(result4.data(), expected4.data()))

        # broadcasting
        y = mn.Tensor([1, 2, 3, 4, 5], dtype="float32")
        lower_small = mn.Tensor([2, 0, 3, 2, 1], dtype="float32")
        upper_small = mn.Tensor([4, 3, 5, 3, 6], dtype="float32")
        result5 = y.clip(lower_small, upper_small)
        expected5 = mn.Tensor([2, 2, 3, 3, 5], dtype="float32")
        # self.assertTrue(np.allclose(result5.data(), expected5.data()))

        # autograd
        z = mn.Tensor([1, 2, 3, 4, 5], dtype="float32", requires_grad=True)
        result6 = z.clip(2, 4)
        # result6.sum().backward()
        expected_grad = mn.Tensor([0, 1, 1, 0, 0], dtype="float32")
        # self.assertTrue(np.allclose(z.grad.data(), expected_grad.data()))

    def test_get_items(self):
        y = mn.ones([3,3])
        y[0, 0] = 10
        self.assertEqual(y[0,0], 10)

    # TODO: different file for each test
    def test_greater_than_scalar(self):
        a = mn.Tensor([0, 0, 1, 0, 1], dtype="float32")
        b = (a > 0)
        self.assertEqual(b[0], 0)
        self.assertEqual(b[1], 0)
        self.assertEqual(b[2], 1)
        self.assertEqual(b[3], 0)
        self.assertEqual(b[4], 1)

    def test_greater_than_tensor(self):
        a = mn.Tensor([0, 1, 2, 3, 4], dtype="float32")
        b = mn.Tensor([2, 2, 2, 2, 2], dtype="float32")
        c = (a > b)
        self.assertEqual(c[0], 0)
        self.assertEqual(c[1], 0)
        self.assertEqual(c[2], 0)
        self.assertEqual(c[3], 1)
        self.assertEqual(c[4], 1)

    def test_greater_equal_scalar(self):
        a = mn.Tensor([0, 1, 2, 1, 0], dtype="float32")
        b = (a >= 1)
        self.assertEqual(b[0], 0)
        self.assertEqual(b[1], 1)
        self.assertEqual(b[2], 1)
        self.assertEqual(b[3], 1)
        self.assertEqual(b[4], 0)

    def test_greater_equal_tensor(self):
        a = mn.Tensor([0, 1, 2, 3, 4], dtype="float32")
        b = mn.Tensor([0, 2, 2, 1, 5], dtype="float32")
        c = (a >= b)
        self.assertEqual(c[0], 1)
        self.assertEqual(c[1], 0)
        self.assertEqual(c[2], 1)
        self.assertEqual(c[3], 1)
        self.assertEqual(c[4], 0)

    def test_less_than_scalar(self):
        a = mn.Tensor([0, 1, 2, 3, 4], dtype="float32")
        b = (a < 2)
        self.assertEqual(b[0], 1)
        self.assertEqual(b[1], 1)
        self.assertEqual(b[2], 0)
        self.assertEqual(b[3], 0)
        self.assertEqual(b[4], 0)

    def test_less_than_tensor(self):
        a = mn.Tensor([0, 1, 2, 3, 4], dtype="float32")
        b = mn.Tensor([1, 1, 3, 2, 4], dtype="float32")
        c = (a < b)
        self.assertEqual(c[0], 1)
        self.assertEqual(c[1], 0)
        self.assertEqual(c[2], 1)
        self.assertEqual(c[3], 0)
        self.assertEqual(c[4], 0)

    def test_less_equal_scalar(self):
        a = mn.Tensor([0, 1, 2, 3, 4], dtype="float32")
        b = (a <= 2)
        self.assertEqual(b[0], 1)
        self.assertEqual(b[1], 1)
        self.assertEqual(b[2], 1)
        self.assertEqual(b[3], 0)
        self.assertEqual(b[4], 0)

    def test_less_equal_tensor(self):
        a = mn.Tensor([0, 1, 2, 3, 4], dtype="float32")
        b = mn.Tensor([0, 0, 2, 4, 3], dtype="float32")
        c = (a <= b)
        self.assertEqual(c[0], 1)
        self.assertEqual(c[1], 0)
        self.assertEqual(c[2], 1)
        self.assertEqual(c[3], 1)
        self.assertEqual(c[4], 0)

    def test_equal_scalar(self):
        a = mn.Tensor([0, 1, 2, 1, 0], dtype="float32")
        b = (a == 1)
        self.assertEqual(b[0], 0)
        self.assertEqual(b[1], 1)
        self.assertEqual(b[2], 0)
        self.assertEqual(b[3], 1)
        self.assertEqual(b[4], 0)

    def test_equal_tensor(self):
        a = mn.Tensor([0, 1, 2, 3, 4], dtype="float32")
        b = mn.Tensor([1, 1, 2, 4, 4], dtype="float32")
        c = (a == b)
        self.assertEqual(c[0], 0)
        self.assertEqual(c[1], 1)
        self.assertEqual(c[2], 1)
        self.assertEqual(c[3], 0)
        self.assertEqual(c[4], 1)

    def test_not_equal_scalar(self):
        a = mn.Tensor([0, 1, 2, 1, 0], dtype="float32")
        b = (a != 1)
        self.assertEqual(b[0], 1)
        self.assertEqual(b[1], 0)
        self.assertEqual(b[2], 1)
        self.assertEqual(b[3], 0)
        self.assertEqual(b[4], 1)

    def test_not_equal_tensor(self):
        a = mn.Tensor([0, 1, 2, 3, 4], dtype="float32")
        b = mn.Tensor([0, 0, 3, 3, 5], dtype="float32")
        c = (a != b)
        self.assertEqual(c[0], 0)
        self.assertEqual(c[1], 1)
        self.assertEqual(c[2], 1)
        self.assertEqual(c[3], 0)
        self.assertEqual(c[4], 1)

    def test_invalid_type(self):
        a = mn.Tensor([0, 1, 2, 3, 4], dtype="float32")
        with self.assertRaises(TypeError):
            b = (a > "invalid")
        with self.assertRaises(TypeError):
            b = (a >= "invalid")
        with self.assertRaises(TypeError):
            b = (a < "invalid")
        with self.assertRaises(TypeError):
            b = (a <= "invalid")
        with self.assertRaises(TypeError):
            b = (a == "invalid")
        with self.assertRaises(TypeError):
            b = (a != "invalid")

    def test_broadcasting(self):
        a = mn.Tensor([[0, 1], [2, 3]], dtype="float32")
        b = (a > 1)
        self.assertEqual(b[0, 0], 0)
        self.assertEqual(b[0, 1], 0)
        self.assertEqual(b[1, 0], 1)
        self.assertEqual(b[1, 1], 1)

    def test_gradient_flow(self):
        a = mn.Tensor([0.0, 1.0, 2.0], dtype="float32", requires_grad=True)
        c = (a > 0.5)
        c.backward()

        self.assertEqual(a.grad[0], 0.0)
        self.assertEqual(a.grad[1], 1.0)
        self.assertEqual(a.grad[2], 1.0)

    def test_load_from_file(self):
        a = mn.randn([10, 10], 0, 1, dtype="float64", device="cpu", debug_name="blah")
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "a")
            a.save_to_file(file_name="a", offset=0, directory=tmp_dir)
            b = mn.Tensor.load_from_file(path)

            c = mn.Tensor.__pow__(a - b, 2) < 1e-2

            self.assertEqual(c.sum()[0], 100)

    # run UTs
if __name__ == '__main__':
    unittest.main(verbosity=2)
