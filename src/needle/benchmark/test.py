import time
from ..init import *

def ThreeByThreeMatMulCheck():
    init_start_time = time.time()
    x = Needle.ones([256, 2048])
    y = Needle.ones([2048, 256])
    init_end_time = time.time()

    mat_mul_start_time= time.time()
    z = x @ y
    mat_mul_end_time = time.time()

    init_elapsed_time = init_end_time - init_start_time
    mat_mul_elapsed_time = mat_mul_end_time - mat_mul_start_time
    print(f"Init Time: {init_elapsed_time:.6f} seconds")
    print(f"Execution Time: {mat_mul_elapsed_time:.6f} seconds")


ThreeByThreeMatMulCheck()