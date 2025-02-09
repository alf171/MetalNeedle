import time
import numpy as np
import MetalNeedle

TILE = 128
def benchmark(i):
    init_start_time = time.time()
    x = MetalNeedle.randn(shape=[TILE, TILE], mean=0, std=1, dtype="float32")
    y = MetalNeedle.randn(shape=[TILE, TILE], mean=0, std=1, dtype="float32")
    init_end_time = time.time()
    init_elapsed_time = init_end_time - init_start_time

    mat_mul_start_time = time.time()
    for _ in range(i):
        z = x @ y
    mat_mul_end_time = time.time()
    mat_mul_elapsed_time = mat_mul_end_time - mat_mul_start_time

    return x, y, z, init_elapsed_time, mat_mul_elapsed_time

def check_correctness(x, y, z):
    x_np = np.array(x.tensorData.rawTensor.data, dtype=np.float32).reshape((TILE, TILE))
    y_np = np.array(y.tensorData.rawTensor.data, dtype=np.float32).reshape((TILE, TILE))
    expected_z = np.dot(x_np, y_np)

    for i in range(TILE):
        for j in range(TILE):
            if expected_z[i, j] != z[i,j]:
                print("❌ Correctness Check: FAILED")
                break

    print("✅ Correctness Check: PASSED")

i = 1
x, y, z, init_time, mat_mul_time = benchmark(i)

print(f"Init Time: {init_time:.6f} seconds")
print(f"Matmul Time (repeated {i} times): {mat_mul_time:.6f} seconds")
print(f"Average Matmul Time: {mat_mul_time / i:.6f} seconds")

check_correctness(x, y, z)
