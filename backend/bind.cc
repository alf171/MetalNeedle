#include <pybind11/pybind11.h>
#include "gpu_backend.h"
#include "cpu_backend.h"

namespace py = pybind11;

PYBIND11_MODULE(backend, m) {
    // bind_gpu(m);
    bind_cpu(m);
}