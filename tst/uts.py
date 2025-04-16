import numpy as np
import math
import MetalNeedle as mn
import torch

def assertAlmostEquals(value1, value2):
    epsilon = 1e-5
    assert(value1 - value2 < epsilon)

def ThreeByThreeMatMulCheck():
    x = mn.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = mn.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    z = x @ y

    expected_result = np.dot(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                             np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    assert(z.shape() == [3,3])

    for i in range(3):
        for j in range(3):
            expected_value = expected_result[i, j]
            calculated_value = z[i, j]
            assert calculated_value == expected_value, (
                f"Value mismatch at ({i},{j}): Expected {expected_value}, Got {calculated_value}"
            )

    print("Matmul passed!")


def ScalarOperations():
    x = mn.ones([3, 3])
    z = (x + 3)
    assert(z[1,1] == 4)
    z = (z - 2)
    assert(z[1,1] == 2)
    z = (z * 3)
    assert(z[1,1] == 6)
    z = (z ** 2)
    assert(z[1,1] == 36)
    z = (z / 2)
    assert(z[1,1] == 18)
    print("Scalar Operations passed!")

    z = z.log()
    assertAlmostEquals(z[1,1], math.log(18))

def SlicingOperations():
    x1 = mn.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    y1 = (x1[1,:])
    assert(y1.shape() == [1,3])
    assert(y1[0,0] == 4)
    assert(y1[0,1] == 5)
    assert(y1[0,2] == 6)

    x2 = mn.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    y2 = (x2[:,1])
    assert(y2.shape() == [3,1])
    assert(y2[0,0] == 2)
    assert(y2[1,0] == 5)
    assert(y2[2,0] == 8)

    print("Slicing Operation passed!")

def SumOperation():
    x1 = mn.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x2 = x1.sum(0)
    assert(x2[0] == 12)
    assert(x2[1] == 15)
    assert(x2[2] == 18)
    assert(x2.shape() == [3])

    x3 = mn.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x4 = x3.sum(1)
    assert(x4.shape() == [3])
    assert(x4[0] == 6)
    assert(x4[1] == 15)
    assert(x4[2] == 24)

    x5 = mn.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x6 = x5.sum([0,1])
    assert(x6[0] == ((9*10)/2))
    assert(x6.shape() == [1])

    print("Sum Operation passed!")

def BroadcastOperation():
    x1 = mn.Tensor([1, 2, 3])
    x2 = x1.reshape([1,3])
    x3 = x2.broadcast([3,3])
    for i in range(3):
        assert(x3[i,0] == 1)
        assert(x3[i,1] == 2)
        assert (x3[i,2] == 3)

    print("Broadcast Operation passed!")

def ReshapeOperations():
    x = mn.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x = x.transpose()
    assert(x[0,0] == 1)
    assert(x[0,1] == 4)
    assert(x[0,2] == 7)
    assert(x[2,2] == 9)
    x = mn.Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    x = x.reshape([9, 1])
    assert(x[0,0] == 1)
    assert(x[8,0] == 3)
    x = x.transpose()
    assert(x[0,8] == 3)

    print("Reshape Operations passed!")

def Autograd():
    a = mn.Tensor([[1, 2], [3, 4]], requires_grad=True, debug_name="a")
    b = mn.Tensor([[5, 6], [7, 8]], requires_grad=True, debug_name="b")
    c = mn.Tensor([[9, 10], [11, 12]], requires_grad=True, debug_name="c")
    x1 = mn.Tensor.__add__(a, b, debug_name="x1")
    x2 = mn.Tensor.__mul__(x1, c, debug_name="x2")
    y1 = mn.Tensor.__mul__(a, c, debug_name="y1")
    z = mn.Tensor.__add__(x2, y1, debug_name="z")
    z.backward()
    assert(a.grad.data() == [18, 20, 22, 24])
    assert(b.grad.data() == [9, 10, 11, 12])
    assert(c.grad.data() == [7, 10, 13, 16])

    a = mn.ones([5, 3], requires_grad=True, debug_name="a")
    b = mn.ones([3, 5], requires_grad=True, debug_name="a")
    res = a @ b
    res.backward()
    assert(a.grad.shape() == [5,3])
    print("Autograd passes!!")

def Pytorch():
    a = torch.ones([5,3], requires_grad=True)
    b = torch.ones([3,5], requires_grad=True)
    #
    # # Create computational graph
    z = a @ b
    #
    z.backward(torch.ones_like(z))
    #
    # # Print computed gradients
    # print("Computed gradients:")
    # print(a.grad)
    # print(f"b.grad = \n{b.grad}")  # Should be c
    # print(f"c.grad = \n{c.grad}")  # Should be 2a + b


def Maximums():
    a = mn.Tensor([[-1, 14], [3, 4]], requires_grad=True, debug_name="a")
    # assert(a.max()[0] == 14)

    b = mn.Tensor([[2, 13], [9, 3]])
    c = a.maximum(b)
    assert(c[0,0] == 2)
    assert(c[0,1] == 14)
    assert(c[1,0] == 9)
    assert(c[1,1] == 4)

    d = a.maximum(0)
    assert(d[0,0] == 0)

    e = mn.Tensor([[1,1,1], [2,2,2]])
    assert(e.max([0])[0] == 2)
    assert(e.max([0])[1] == 2)
    assert(e.max([0])[2] == 2)
    assert(e.max([1])[0] == 1)
    assert(e.max([1])[1] == 2)

    print("Maximum test passed!")

# run UTs
if __name__ == "__main__":
    ThreeByThreeMatMulCheck()
    ScalarOperations()
    SlicingOperations()
    SumOperation()
    BroadcastOperation()
    ReshapeOperations()
    Autograd()
    # Pytorch()
    Maximums()