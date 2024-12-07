#ifndef GPU_BINDINGS_H
#define GPU_BINDINGS_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_gpu(py::module &m);

#endif