#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cpu/backend.h"
#include "metal/backend.h"

PYBIND11_MODULE(backend, m) {
    bind_cpu(m);
    bind_metal(m);
}
