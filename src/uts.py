import numpy as np
import math
import MetalNeedle
import torch

def assertAlmostEquals(value1, value2):
    epsilon = 1e-5
    assert(value1 - value2 < epsilon)

# TODO: use pytest
def ThreeByThreeMatMulCheck():
    x = MetalNeedle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = MetalNeedle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
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
    x = MetalNeedle.ones([3, 3])
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
    x1 = MetalNeedle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    y1 = (x1[1,:])
    assert(y1[0] == 4)
    assert(y1[1] == 5)
    assert(y1[2] == 6)

    x2 = MetalNeedle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    y2 = (x2[:,1])
    assert(y2[0] == 2)
    assert(y2[1] == 5)
    assert(y2[2] == 8)

    print("Slicing Operation passed!")

def SumOperation():
    x1 = MetalNeedle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x2 = x1.sum(0)
    assert(x2[0] == 12)
    assert(x2[1] == 15)
    assert(x2[2] == 18)
    assert(x2.shape() == [3])

    x3 = MetalNeedle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x4 = x3.sum(1)
    assert(x4.shape() == [3])
    assert(x4[0] == 6)
    assert(x4[1] == 15)
    assert(x4[2] == 24)

    x5 = MetalNeedle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x6 = x5.sum([0,1])
    assert(x6[0] == ((9*10)/2))
    assert(x6.shape() == [1])

    print("Sum Operation passed!")

def BroadcastOperation():
    x1 = MetalNeedle.Tensor([1, 2, 3])
    x2 = x1.reshape([1,3])
    print(x2)
    x3 = x2.broadcast([3,3])
    print(x3)
    for i in range(3):
        assert(x3[i,0] == 1)
        assert(x3[i,1] == 2)
        assert (x3[i,2] == 3)

    print("Broadcast Operation passed!")

def ReshapeOperations():
    x = MetalNeedle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x = x.transpose()
    assert(x[0,0] == 1)
    assert(x[0,1] == 4)
    assert(x[0,2] == 7)
    assert(x[2,2] == 9)
    x = MetalNeedle.Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    x = x.reshape([9, 1])
    assert(x[0,0] == 1)
    assert(x[8,0] == 3)
    x = x.transpose()
    assert(x[0,8] == 3)

    print("Reshape Operations passed!")

def Autograd():
    a = MetalNeedle.Tensor([[1, 2], [3, 4]], requires_grad=True, debug_name="a")
    b = MetalNeedle.Tensor([[5, 6], [7, 8]], requires_grad=True, debug_name="b")
    c = MetalNeedle.Tensor([[9, 10], [11, 12]], requires_grad=True, debug_name="c")
    x1 = MetalNeedle.Tensor.__add__(a, b, debug_name="x1")
    x2 = MetalNeedle.Tensor.__mul__(x1, c, debug_name="x2")
    y1 = MetalNeedle.Tensor.__mul__(a, c, debug_name="y1")
    z = MetalNeedle.Tensor.__add__(x2, y1, debug_name="z")
    z.backward()
    assert(a.grad.data() == [18, 20, 22, 24])
    assert(b.grad.data() == [9, 10, 11, 12])
    assert(c.grad.data() == [7, 10, 13, 16])

    a = MetalNeedle.ones([5, 3], requires_grad=True, debug_name="a")
    b = MetalNeedle.ones([3, 5], requires_grad=True, debug_name="a")
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


def MetalAddTest():
    x = MetalNeedle.ones([32, 32], device="metal", dtype='float32', debug_name="x")
    y = MetalNeedle.ones([32, 32], device="metal", dtype='float32', debug_name="y")
    z = x + y
    assert(z[15,2] == 2)
    print("Metal add passed!")


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
    MetalAddTest()