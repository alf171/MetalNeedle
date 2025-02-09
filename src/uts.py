import numpy as np
import MetalNeedle

# TODO: use pytest
def ThreeByThreeMatMulCheck():
    x = MetalNeedle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = MetalNeedle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    z = x @ y

    expected_result = np.dot(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                             np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    assert(z.tensorData.shape() == [3,3])

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
    x1.sum(0)
    assert(x1[0] == 12)
    assert(x1[1] == 15)
    assert(x1[2] == 18)
    assert(x1.tensorData.shape() == [3])

    x2 = MetalNeedle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x2.sum(1)
    assert(x2.tensorData.shape() == [3])
    assert(x2[0] == 6)
    assert(x2[1] == 15)
    assert(x2[2] == 24)

    x3 = MetalNeedle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x3.sum([0,1])
    assert(x3[0] == ((9*10)/2))
    assert(x3.tensorData.shape() == [1])

    print("Sum Operation passed!")

def BroadcastOperation():
    x1 = MetalNeedle.Tensor([1, 2, 3])
    x1.reshape([1,3])
    x1.broadcast([3,3])
    for i in range(3):
        assert(x1[i,0] == 1)
        assert(x1[i,1] == 2)
        assert (x1[i,2] == 3)

    print("Broadcast Operation passed!")

def ReshapeOperations():
    x = MetalNeedle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x.transpose()
    assert(x[0] == 1)
    assert(x[1] == 4)
    assert(x[2] == 7)
    assert(x[2,2] == 9)

    x = MetalNeedle.Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    x.reshape([9, 1])
    assert(x[0,0] == 1)
    assert(x[8,0] == 3)
    # print(x.tensorData.shape)
    x.transpose()
    x.tensorData.shape()[0] = 3
    # print(x.tensorData.shape)
    # assert(x[0,8] == 3)

    print("Reshape Operations passed!")

def Autograd():
    x = MetalNeedle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], requires_grad=True)
    y = x + x
    # (y.grad_fn.tensor(4))
    # assert(x.grad == 8)

    z = y * y
    z.grad_fn(4)

def test():
    x = MetalNeedle.Tensor([1,2,3], device="metal")
    # print(x.tensorData.ones_like().data())


# run UTs
# ThreeByThreeMatMulCheck()
# ScalarOperations()
# SlicingOperations()
# SumOperation()
# BroadcastOperation()
# ReshapeOperations()
# Autograd()
test()