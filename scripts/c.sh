clang++ -g -O3 -Wall -shared -std=c++17 -fPIC \
    -I/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/include/python3.13 \
    -I/opt/homebrew/lib/python3.13/site-packages/pybind11/include \
    -L/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/lib \
    -lpython3.13 \
    -o tmp/backend$(python3.13-config --extension-suffix) \
    src/backend/bind.cc src/backend/cpu_backend.cc src/backend/gpu_backend.cc