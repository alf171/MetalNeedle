import time
import Needle  # Assuming the Needle library is implemented correctly.

def benchmark():
    init_start_time = time.time()
    x = Needle.randn(shape=[2048, 2048], mean=5, std=1, dtype="int32")
    y = Needle.randn(shape=[2048, 2048], mean=5, std=1, dtype="int32")
    init_end_time = time.time()

    mat_mul_start_time = time.time()
    z = x @ y
    mat_mul_end_time = time.time()

    init_elapsed_time = init_end_time - init_start_time
    mat_mul_elapsed_time = mat_mul_end_time - mat_mul_start_time

    return init_elapsed_time, mat_mul_elapsed_time

# Execute the benchmark 10 times and calculate averages
init_times = []
mat_mul_times = []

for _ in range(1):
    init_time, mat_mul_time = benchmark()
    init_times.append(init_time)
    mat_mul_times.append(mat_mul_time)

average_init_time = sum(init_times) / len(init_times)
average_mat_mul_time = sum(mat_mul_times) / len(mat_mul_times)

print(f"Init Time: {average_init_time:.6f} seconds")
print(f"Execution Time: {average_mat_mul_time:.6f} seconds")
