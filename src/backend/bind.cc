#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "gpu_backend.h"
#include "cpu_backend.h"

namespace py = pybind11;

void bind_gpu(py::module &m) {
    auto metal = m.def_submodule("metal");
    // Bind operations
//    bind_operations(metal, "simple_shader");
}

PYBIND11_MODULE(backend, m) {
    // bind_gpu(m);
    bind_cpu(m);
}