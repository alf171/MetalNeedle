#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"

namespace py = pybind11;

template<typename T>
class MetalBackend {
public:
    MetalBackend() = default;
    device = MTL::CreateSystemDefaultDevice();
    commandQueue = device->newCommandQueue();

    NS::Error* error = nullptr;
    NS::String* shaderPath = NS::String::string("tensor.metallib", NS::UTF8StringEncoding);
    opLibrary = device->newLibrary(shaderPath, nullptr, &error);
    if (!opLibrary) {
        std::cerr << "Failed to load Metal library: " << error->localizedDescription()->utf8String() << std::endl;
        exit(1);
    }

};

void bind_metal(py::module &m) {
    auto metal = m.def_submodule("metal");
    metal.def("metal_test", []() { return "Metal backend loaded!"; });
}
