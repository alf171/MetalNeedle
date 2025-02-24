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
    Tensor<T> ewise_add(Tensor<T>& e1, Tensor<T>& e2);
//    Tensor<T> ewise_sub(Tensor<T>& e1, Tensor<T>& e2);
//    Tensor<T> ewise_exp(Tensor<T>& e1, float v1);
//    Tensor<T> ewise_div(Tensor<T>& e1, Tensor<T>& e2);
//    Tensor<T> ewise_mul(Tensor<T>& e1, Tensor<T>& e2);
//    Tensor<T> tiled_mat_mul(Tensor<T>& e1, Tensor<T>& e2);
//    Tensor<T> scalar_add(Tensor<T>& tensor, T scalar);
//    Tensor<T> scalar_sub(Tensor<T>& tensor, T scalar);
//    Tensor<T> scalar_mul(Tensor<T>& tensor, T scalar);
//    Tensor<T> scalar_div(Tensor<T>& tensor, T scalar);
//    Tensor<T> scalar_exp(Tensor<T>& tensor, T scalar);
};

void bind_metal(py::module &m);

#endif