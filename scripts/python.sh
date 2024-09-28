#!/bin/bash

# put __pycache__ into tmp dir
CACHE_DIR=$pwd
export PYTHONPYCACHEPREFIX="$CACHE_DIR/tmp"

# run testing script
python3 -m Needle.testing