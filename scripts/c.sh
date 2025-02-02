clang++ -O3 -march=native -ftree-vectorize -funroll-loops -flto \
    -Wall -shared -std=c++23 -fPIC -frtti \
    -I/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/include/python3.13 \
    -I/opt/homebrew/lib/python3.13/site-packages/pybind11/include \
    -I/opt/homebrew/opt/libomp/include \
    -Isrc/backend/tensor.h \
    -L/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/lib \
    -L/opt/homebrew/opt/libomp/lib \
    -Xpreprocessor -fopenmp \
    -lpython3.13 -lomp \
    -o tmp/backend$(python3.13-config --extension-suffix) \
    src/backend/bind.cc src/backend/cpu_backend.cc
