#!/bin/bash

export PYTHONPYCACHEPREFIX="tmp/pycache"

# run testing script
python3.13 -m scripts.setup install --build "tmp/build" --prefix "tmp/dist" > /dev/null 2>&1
