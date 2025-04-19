#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(pwd)"

source ./scripts/setup.sh
# run testing script
python3.13 -m tst.test_tensor
#python3.13 -m tst.metal
#python3.13 -m tst.nn
#python3.13 -m tst.mnsit
