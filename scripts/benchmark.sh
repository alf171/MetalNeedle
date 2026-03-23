#!/bin/bash

source ./scripts/setup.sh
# needle speed
python3.14 -m tst.needle_matmul

# numpy speed
#python3.14 -m tst.benchmark.numpy_matmul
