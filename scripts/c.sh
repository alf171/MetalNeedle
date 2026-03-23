#!/bin/bash

set -euo pipefail

PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python3.14}
PYBIND11_DIR=${PYBIND11_DIR:-$(brew --prefix pybind11)/share/cmake/pybind11}

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DPython3_EXECUTABLE="$PYTHON_EXECUTABLE" \
  -Dpybind11_DIR="$PYBIND11_DIR" \
  ${EXTRA_CMAKE_ARGS:-}

cmake --build build

cp build/compile_commands.json compile_commands.json
