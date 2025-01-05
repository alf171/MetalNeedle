import numpy as np
import Needle

def ThreeByThreeMatMulCheck():
    x = Needle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = Needle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    z = x @ y

    expected_result = np.dot(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                             np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    assert(z.shape == [3,3])

    for i in range(3):
        for j in range(3):
            expected_value = expected_result[i, j]
            calculated_value = z[i, j]
            assert calculated_value == expected_value, (
                f"Value mismatch at ({i},{j}): Expected {expected_value}, Got {calculated_value}"
            )

    print("ThreeByThreeMatMulCheck passed!")


def ScalarOperations():
    x = Needle.ones([3,3])
    z = (x + 3)

    assert(z[0,0] == 4)
    assert(z[1,0] == 4)
    assert(z[0,2] == 4)

    print("ScalarOperations passed!")

def SlicingOperations():
    x1 = Needle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    y1 = (x1[1,:])
    assert(y1[0] == 4)
    assert(y1[1] == 5)
    assert(y1[2] == 6)

    x2 = Needle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    y2 = (x2[:,1])
    assert(y2[0] == 2)
    assert(y2[1] == 5)
    assert(y2[2] == 8)

    print("SlicingOperation passed!")

def SumOperation():
    x1 = Needle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x1.sum(0)
    assert(x1[0] == 12)
    assert(x1[1] == 15)
    assert(x1[2] == 18)
    assert(x1.shape == [3])

    x2 = Needle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x2.sum(1)
    assert(x2.shape == [3])
    assert(x2[0] == 6)
    assert(x2[1] == 15)
    assert(x2[2] == 24)

    x3 = Needle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    x3.sum([0,1])
    assert(x3[0] == ((9*10)/2))
    assert(x3.shape == [1])

    print("SumOperation passed!")

# run UTs
ThreeByThreeMatMulCheck()
ScalarOperations()
SlicingOperations()
SumOperation()