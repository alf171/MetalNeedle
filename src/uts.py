import numpy as np
import math
import MetalNeedle

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
    # Test case 1: Basic addition and gradient propagation
    a = MetalNeedle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], requires_grad=True, debug_name="a")
    b = MetalNeedle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], requires_grad=True, debug_name="b")
    c = MetalNeedle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], requires_grad=True, debug_name="c")
    d = MetalNeedle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], requires_grad=True, debug_name="d")
    # e = MetalNeedle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], requires_grad=True, debug_name="c")
    w = MetalNeedle.Tensor.__add__(a, b, 'w')
    x = MetalNeedle.Tensor.__add__(c, d, 'x')
    y = MetalNeedle.Tensor.__mul__(w, x, 'y')
    # z = y + e

    y.backward()

    # dz/dx = 8x, so grad should be 8 times the original tensor
    print(x.grad.data())

    # Test case 2: More complex operations
    # a = MetalNeedle.Tensor([2.0], requires_grad=True)
    # b = MetalNeedle.Tensor([3.0], requires_grad=True)
    #
    # # Forward pass: c = a * b, d = a + c, e = d / b
    # c = a * b  # c = 6
    # d = a + c  # d = 2 + 6 = 8
    # e = d / b  # e = 8/3
    #
    # # Backward pass
    # e.backward()
    #
    # # Gradient calculations:
    # # de/da = (1 + b) / b = (1 + 3) / 3 = 4/3
    # # de/db = -a*(a + a*b)/(b*b) = -2*(2 + 2*3)/(3*3) = -2*8/9 = -16/9
    #
    # assertAlmostEquals(a.grad.tensorData.rawTensor[0], 4/3)
    # assertAlmostEquals(b.grad.tensorData.rawTensor[0], -16/9)
    #
    # # Test case 3: Multiple backward passes (gradient accumulation)
    # p = MetalNeedle.Tensor([1.0], requires_grad=True)
    # q = p * p  # q = p²
    #
    # # First backward pass
    # q.backward()
    # # dq/dp = 2p = 2
    # assertAlmostEquals(p.grad.tensorData.rawTensor[0], 2.0)
    #
    # # Second backward pass should accumulate
    # q.backward()
    # # Now total gradient should be 4
    # assertAlmostEquals(p.grad.tensorData.rawTensor[0], 4.0)
    #
    # # Test case 4: Zero_grad functionality
    # optimizer = MetalNeedle.SGDOptimizer([p], learning_rate=0.1)
    # optimizer.zero_grad()
    #
    # assert p.grad is None, "Gradient should be None after zero_grad"
    #
    # # Test case 5: Optimizer step
    # p = MetalNeedle.Tensor([1.0], requires_grad=True)
    # q = p * p  # q = p²
    # q.backward()  # dq/dp = 2p = 2
    #
    # optimizer = MetalNeedle.SGDOptimizer([p], learning_rate=0.1)
    # optimizer.step()
    #
    # # New value should be: 1.0 - 0.1 * 2.0 = 0.8
    # assertAlmostEquals(p.tensorData.rawTensor[0], 0.8)
    #
    # print("All autograd tests passed!")

def MetalAddTest():
    x = MetalNeedle.ones([32, 32], device="metal", dtype='float32', _debug_name="x")
    y = MetalNeedle.ones([32, 32], device="metal", dtype='float32', _debug_name="y")
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
    MetalAddTest()