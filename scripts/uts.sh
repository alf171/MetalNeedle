#!/bin/bash

source ./scripts/setup.sh
# run testing script
python3.14 -m tst.test_tensor
python3.14 -m tst.metal
python3.14 -m tst.nn
