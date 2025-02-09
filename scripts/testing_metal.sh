clang++ -Wall -std=c++17 -I./metal-cpp -I./metal-cpp-extensions -fno-objc-arc -O2 \
    -framework Metal -framework Foundation -framework Cocoa -framework CoreGraphics -framework MetalKit \
    -I src/backend/metal \
    -I/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/include/python3.13 \
    -I/opt/homebrew/lib/python3.13/site-packages/pybind11/include \
    -Isrc/backend \
    src/backend/main.cc
