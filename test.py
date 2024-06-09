import torch
import time
# enable metal 
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

MSIZE = 4096

def test_metal():
    a = torch.randn((MSIZE, MSIZE), device=DEVICE)
    b = torch.randn((MSIZE, MSIZE), device=DEVICE)
    return a @ b

def test_cpu():
    a = torch.randn((MSIZE, MSIZE))
    b = torch.randn((MSIZE, MSIZE))
    return a @ b

if __name__ == "__main__":
    
    start_time_metal = time.time()
    result = test_metal()
    end_time_metal = time.time()

    start_time_cpu = time.time()
    result = test_metal()
    end_time_cpu = time.time()

    # print(result)
    print(f'METAL RUNTIME {end_time_metal - start_time_metal} seconds')
    print(f'CPU RUNTIME {end_time_cpu - start_time_cpu} seconds')

