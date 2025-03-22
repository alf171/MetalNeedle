#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include <unistd.h>
#include "backend.h"

namespace py = pybind11;

template<typename T>
MTL::Device* MetalBackend<T>::device = nullptr;

template<typename T>
MTL::CommandQueue* MetalBackend<T>::commandQueue = nullptr;

template<typename T>
MTL::Library* MetalBackend<T>::opLibrary = nullptr;

// Explicit instantiation for the types
template class MetalBackend<int>;
template class MetalBackend<float>;
template class MetalBackend<double>;
template class MetalBackend<long long>;

template<typename T>
MetalBackend<T>::MetalBackend() {
    if (!device) {
        device = MTL::CreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Failed to load device" << std::endl;
            exit(1);
        }
        commandQueue = device->newCommandQueue();
        if (!commandQueue) {
            std::cerr << "Failed to create command queue" << std::endl;
            exit(1);
        }
        NS::Error* error = nullptr;
        auto filepath = NS::String::string("./tmp/metal_backend.metallib", NS::ASCIIStringEncoding);
        opLibrary = device->newLibrary(filepath, &error);
        if (!opLibrary) {
            std::cerr << "Failed to load Metal library: " << error->localizedDescription()->utf8String() << std::endl;
            exit(1);
        }
    }
};

template<typename T>
MetalTensor<T> MetalBackend<T>::ewise_add(MetalTensor<T>& e1, MetalTensor<T>& e2) {
    if (e1.shape != e2.shape) {
        throw std::invalid_argument("Tensors must have same shapes for ewise operations");
    }

    size_t num_elements = e1.cpu_data.size();
    MTL::Buffer* buffer_e1 = device->newBuffer(num_elements * sizeof(T), MTL::ResourceStorageModeShared);
    MTL::Buffer* buffer_e2 = device->newBuffer(num_elements * sizeof(T), MTL::ResourceStorageModeShared);
    std::memcpy(buffer_e1->contents(), e1.cpu_data.data(), num_elements * sizeof(T));
    std::memcpy(buffer_e2->contents(), e2.cpu_data.data(), num_elements * sizeof(T));

    MTL::Buffer* buffer_res = device->newBuffer(num_elements * sizeof(T), MTL::ResourceStorageModeShared);

    NS::Error* error = nullptr;
    auto str = NS::String::string("metal_ewise_add", NS::ASCIIStringEncoding);
    MTL::Function* computeFunction = opLibrary->newFunction(str);
    if (computeFunction == nullptr) {
        std::cerr << "Failed to find function 'metal_ewise_add' in the Metal library" << std::endl;
        exit(1);
    }

    MTL::ComputePipelineState* pipelineState = device->newComputePipelineState(computeFunction, &error);
    computeFunction->release();

    if (error) {
        std::cerr << "Failed to create pipeline state: " << error->localizedDescription()->utf8String() << std::endl;
        exit(1);
    }

    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();

    encoder->setComputePipelineState(pipelineState);
    encoder->setBuffer(buffer_e1, 0, 0);
    encoder->setBuffer(buffer_e2, 0, 1);
    encoder->setBuffer(buffer_res, 0, 2);

    MTL::Size gridSize = MTL::Size::Make(num_elements, 1, 1);
    NS::UInteger threadGroupSize = pipelineState->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > num_elements) {
        threadGroupSize = num_elements;
    }
    MTL::Size safeThreadSize = MTL::Size::Make(threadGroupSize, 1, 1);

    encoder->dispatchThreads(gridSize, safeThreadSize);
    encoder->endEncoding();

    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    std::vector<T> res(static_cast<T*>(buffer_res->contents()), 
                      static_cast<T*>(buffer_res->contents()) + e1.cpu_data.size());

    buffer_e1->release();
    buffer_e2->release();
    buffer_res->release();
    pipelineState->release();

    return MetalTensor<T>::initialize(res, e1.shape);
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
    bind_metal_tensor<int32_t>(metal, "IntTensor");
    bind_metal_tensor<int64_t>(metal, "LongTensor");
    bind_metal_tensor<float>(metal, "FloatTensor");
    bind_metal_tensor<double>(metal, "DoubleTensor");
}
