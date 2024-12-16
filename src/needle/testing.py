import numpy as np
from .init import *

def ThreeByThreeMatMulCheck():

    # Initialize input matrices
    x = Needle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = Needle.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    z = x @ y

    # Expected result
    expected_result = np.dot(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                             np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    # Debugging helper
    def debug_matrix(matrix, name):
        print(f"\n{name} matrix:")
        for i in range(len(matrix)):
            print(matrix[i])

    expected_flat = expected_result.flatten()
    calculated_flat = [z.get_item([i, j]) for i in range(3) for j in range(3)]

    # print("\nExpected Result (Flattened):", expected_flat)
    # print("Calculated Result (Flattened):", calculated_flat)

    for i in range(3):
        for j in range(3):
            expected_value = expected_result[i, j]
            calculated_value = z.get_item([i, j])
            assert calculated_value == expected_value, (
                f"Value mismatch at ({i},{j}): Expected {expected_value}, Got {calculated_value}"
            )

    # debug_matrix(expected_result.tolist(), "Expected")
    # debug_matrix([[z.get_item([i, j]) for j in range(3)] for i in range(3)], "Calculated")

    print("ThreeByThreeMatMulCheck passed!")


def ScalarOperations():
    x = Needle.ones([3,3])
    z = (x + 3)

    assert(z.get_item([0,0]) == 4)
    assert(z.get_item([1,0]) == 4)
    assert(z.get_item([0,2]) == 4)

ThreeByThreeMatMulCheck()
ScalarOperations()