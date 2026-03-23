#ifndef METAL_BACKEND_H
#define METAL_BACKEND_H

#include "tensor.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Metal/Metal.hpp>

namespace py = pybind11;

template <typename T> class MetalBackend {
public:
  MetalBackend();
  static MTL::Device *device;
  static MTL::CommandQueue *commandQueue;
  static MTL::Library *opLibrary;
  // operations
  MetalTensor<T> ewise_add(MetalTensor<T> &e1, MetalTensor<T> &e2);
  MetalTensor<T> scalar_mul(MetalTensor<T> &tensor, T scalar);
  MetalTensor<T> mat_mul(MetalTensor<T> &e1, MetalTensor<T> &e2);

private:
  MTL::ComputePipelineState *make_pipeline(const char *function_name);
  MTL::Size make_threadgroup_size(MTL::ComputePipelineState *pipeline_state,
                                  size_t num_elements);
  MetalTensor<T> run_binary_kernel(const char *function_name,
                                   MetalTensor<T> &e1, MetalTensor<T> &e2);
  MetalTensor<T> run_scalar_kernel(const char *function_name,
                                   MetalTensor<T> &tensor, T scalar);
};

void bind_metal(py::module &m);

#endif
