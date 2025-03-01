#!/bin/bash

export PYTHONPYCACHEPREFIX="tmp/pycache"

# run testing script
python3.13 -m setup install > /dev/null 2>&1
