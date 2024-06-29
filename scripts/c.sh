clang++ -O3 -Wall -shared -std=c++17 -fPIC \
    -I/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/include/python3.12 \
    -I/opt/homebrew/Cellar/pybind11/2.12.0/libexec/lib/python3.12/site-packages/pybind11/include \
    -L/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/lib \
    -lpython3.12 \
    -o cpu_backend.so cpu_backend.cc
