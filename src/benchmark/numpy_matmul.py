import time
import numpy as np
# print(np.__config__.show())

def benchmark(i):
    init_start_time = time.time()
    x = np.random.randn(4096, 4096).astype(np.float32)
    y = np.random.randn(4096, 4096).astype(np.float32)
    init_end_time = time.time()

    init_elapsed_time = init_end_time - init_start_time

    mat_mul_start_time = time.time()
    for _ in range(i):
        z = np.dot(x, y)
    mat_mul_end_time = time.time()

    mat_mul_elapsed_time = mat_mul_end_time - mat_mul_start_time

    return init_elapsed_time, mat_mul_elapsed_time

i = 1
init_time, mat_mul_time = benchmark(i)


print(f"Init Time: {init_time:.6f} seconds")
print(f"Matmul Time (repeated {i} times): {mat_mul_time:.6f} seconds")
print(f"Average Matmul Time: {mat_mul_time / i:.6f} seconds")
