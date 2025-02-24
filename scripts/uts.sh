#!/bin/bash

# export DYLD_PRINT_LIBRARIES=1
# export DYLD_PRINT_STATISTICS=1

source ./scripts/setup.sh
# run testing script
python3.13 -m src.uts
