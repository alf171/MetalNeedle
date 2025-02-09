#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include "backend.h"

namespace py = pybind11;

template<typename T>
MetalBackend<T>::MetalBackend() {
    device = MTL::CreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Failed to load device" << std::endl;
        exit(1);
    }

    commandQueue = device->newCommandQueue();

    NS::Error* error = nullptr;
    NS::String* shaderPath = NS::String::string("backend.metal", NS::UTF8StringEncoding);
    opLibrary = device->newLibrary(shaderPath, nullptr, &error);
    if (!opLibrary) {
        std::cerr << "Failed to load Metal library: " << error->localizedDescription()->utf8String() << std::endl;
        exit(1);
    }
};

template<typename T>
Tensor<T> MetalBackend<T>::ewise_add(Tensor<T>& e1, Tensor<T>& e2) {
    if (e1.shape != e2.shape) {
        throw std::invalid_argument("Tensors must have same shapes for ewise operations");
    }

    MTL::Buffer *buffer_e1 = device->newBuffer(e1.data.size() * sizeof(T), MTL::ResourceStorageModeShared);
    std::memcpy(buffer_e1->contents(), e1.data.data(), e1.data.size() * sizeof(T));
    MTL::Buffer *buffer_e2 = device->newBuffer(e2.data.size() * sizeof(T), MTL::ResourceStorageModeShared);
    std::memcpy(buffer_e2->contents(), e2.data.data(), e2.data.size() * sizeof(T));

    MTL::Buffer *buffer_res = device->newBuffer(e2.data.size(), MTL::ResourceStorageModeShared);

    NS::Error* error = nullptr;
    MTL::Function* computeFunction = opLibrary->newFunction(NS::String::string("metal_ewise_add", NS::UTF8StringEncoding));
    MTL::ComputePipelineState* pipelineState = device->newComputePipelineState(computeFunction, &error);

    if (error) {
        std::cerr << "Failed to create pipeline state: " << error->localizedDescription()->utf8String() << std::endl;
        exit(1);
    }

    // Create command buffer and encoder
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

     // set the encoder for the fn call
     encoder->setBuffer(buffer_e1, 0, 0);
     encoder->setBuffer(buffer_e2, 1, 0);
     encoder->setBuffer(buffer_res, 2, 0);

     // set the threading
     MTL::Size gridSize(e1.data.size(), 1, 1);
     MTL::Size threadgroupSize(64, 1, 1); // Assume a 64-thread block
     encoder->dispatchThreads(gridSize, threadgroupSize);

     encoder->endEncoding();
     commandBuffer->commit();
     commandBuffer->waitUntilCompleted();

     T* result_data = static_cast<T*>(buffer_res->contents());
     std::vector<T> res(result_data, result_data + e1.data.size());
     return Tensor<T>::initialize(res, e1.shape);
}

//template<typename T>
//void MetalAdder::sendComputeCommand() {
//    // Create a command buffer to hold commands.
//    MTL::CommandBuffer *commandBuffer = _mCommandQueue->commandBuffer();
//    assert(commandBuffer != nullptr);
//
//    // Start a compute pass.
//    MTL::ComputeCommandEncoder *computeEncoder = commandBuffer->computeCommandEncoder();
//    assert(computeEncoder != nullptr);
//
//    encodeAddCommand(computeEncoder);
//
//    // End the compute pass.
//    computeEncoder->endEncoding();
//
//    // Execute the command.
//    commandBuffer->commit();
//
//    // Normally, you want to do other work in your app while the GPU is running,
//    // but in this example, the code simply blocks until the calculation is complete.
//    commandBuffer->waitUntilCompleted();
//}

template <typename T>
void bind_metal_operations(pybind11::module& m, const std::string& class_name) {
    py::class_<MetalBackend<T>>(m, class_name.c_str())
        .def(py::init<>())
        .def("ewise_add", &MetalBackend<T>::ewise_add);
}

void bind_metal(py::module &m) {
    auto metal = m.def_submodule("metal");
    // operations
    bind_metal_operations<int32_t>(metal, "IntOperation");
    bind_metal_operations<int64_t>(metal, "LongOperation");
    bind_metal_operations<float>(metal, "FloatOperation");
    bind_metal_operations<double>(metal, "DoubleOperation");
    // data
    bind_tensor<int32_t>(metal, "IntTensor");
    bind_tensor<int64_t>(metal, "LongTensor");
    bind_tensor<float>(metal, "FloatTensor");
    bind_tensor<double>(metal, "DoubleTensor");
}
