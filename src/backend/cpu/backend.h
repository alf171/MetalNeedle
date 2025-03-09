#ifndef CPU_BACKEND_H
#define CPU_BACKEND_H

#include <algorithm>
#include <chrono>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>
#include "tensor.h"

namespace py = pybind11;

template<typename T>
class CPUBackend {
public:
    CPUBackend() = default;

    Tensor<T> ewise_add(Tensor<T>& e1, Tensor<T>& e2);
    Tensor<T> ewise_sub(Tensor<T>& e1, Tensor<T>& e2);
    Tensor<T> ewise_exp(Tensor<T>& e1, float v1);
    Tensor<T> ewise_div(Tensor<T>& e1, Tensor<T>& e2);
    Tensor<T> ewise_mul(Tensor<T>& e1, Tensor<T>& e2);
    Tensor<T> tiled_mat_mul(Tensor<T>& e1, Tensor<T>& e2);
    Tensor<T> scalar_add(Tensor<T>& tensor, T scalar);
    Tensor<T> scalar_sub(Tensor<T>& tensor, T scalar);
    Tensor<T> scalar_mul(Tensor<T>& tensor, T scalar);
    Tensor<T> scalar_div(Tensor<T>& tensor, T scalar);
    Tensor<T> scalar_exp(Tensor<T>& tensor, T scalar);
    Tensor<T> log(Tensor<T>& tensor);
    Tensor<T> sum(Tensor<T>& tensor, bool keepDims);

private:
    void tile_compute(const std::vector<T>& e1, const std::vector<T>& e2, std::vector<T>& res,
                      size_t block_x, size_t block_y, size_t e1_cols, size_t e2_cols, size_t e1_rows);
};

void bind_cpu(py::module &m);

#endif