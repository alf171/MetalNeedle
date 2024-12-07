#ifndef CPU_BINDINGS_H
#define CPU_BINDINGS_H

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_cpu(py::module &m);

#endif