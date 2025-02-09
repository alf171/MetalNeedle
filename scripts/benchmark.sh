#!/bin/bash

source ./scripts/setup.sh
# run testing script
python3.13 -m src.needle_matmul
#python3.13 -m src.benchmark.numpy_matmul
