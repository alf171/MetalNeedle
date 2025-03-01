#ifndef METAL_BACKEND_H
#define METAL_BACKEND_H

#include <algorithm>
#include <chrono>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h"

#include <Metal/Metal.hpp>

namespace py = pybind11;

template<typename T>
class MetalBackend {
public:
    MetalBackend();
    static MTL::Device* device;
    static MTL::CommandQueue* commandQueue;
    static MTL::Library* opLibrary;
    // operations
    MetalTensor<T> ewise_add(MetalTensor<T>& e1, MetalTensor<T>& e2);
};

void bind_metal(py::module &m);

#endif