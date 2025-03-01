clang++ -g -arch arm64 -march=native -ftree-vectorize -funroll-loops -flto \
    -Wall -shared -std=c++23 -fPIC -frtti \
    -I/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/include/python3.13 \
    -I/opt/homebrew/lib/python3.13/site-packages/pybind11/include \
    -I/opt/homebrew/opt/libomp/include \
    -Isrc/backend/cpu \
    -I./metal-cpp \
    -L/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/lib \
    -L/opt/homebrew/opt/libomp/lib \
    -Xpreprocessor -fopenmp \
    -lpython3.13 -lomp \
    -framework Metal -framework Foundation -framework MetalKit \
    -stdlib=libc++ -fno-objc-arc -o tmp/backend$(python3.13-config --extension-suffix) \
    src/backend/bind.cc src/backend/cpu/backend.cc src/backend/metal/backend.cc
