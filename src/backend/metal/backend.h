#pragma once

#include <algorithm>
#include <chrono>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"

namespace py = pybind11;

template<typename T>
class MetalBackend {
public:
    MetalBackend() = default;

private:
    MTL::Device* device;
    MTL::CommandQueue* commandQueue;
    MTL::Library* opLibrary;
};

void bind_metal(py::module &m);
