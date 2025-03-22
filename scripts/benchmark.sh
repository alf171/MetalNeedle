#!/bin/bash

source ./scripts/setup.sh
# needle speed
python3.13 -m tst.needle_matmul

# numpy speed
#python3.13 -m tst.benchmark.numpy_matmul
