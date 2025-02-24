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

// Explicit instantiation of the static members for each type
template<typename T>
MTL::Device* MetalBackend<T>::device = nullptr;

template<typename T>
MTL::CommandQueue* MetalBackend<T>::commandQueue = nullptr;

template<typename T>
MTL::Library* MetalBackend<T>::opLibrary = nullptr;

// Explicit instantiation for the types you're using
template class MetalBackend<int>;
template class MetalBackend<float>;
template class MetalBackend<double>;
template class MetalBackend<long long>;

template<typename T>
auto copyToBuffer = [](MTL::Device* device, const std::vector<T>& data, const char* bufferName) -> MTL::Buffer* {
    size_t byteSize = data.size() * sizeof(T);
    std::cout << "Creating buffer for " << bufferName << " with size: " << byteSize << std::endl;
    
    MTL::Buffer* buffer = device->newBuffer(byteSize, MTL::ResourceStorageModeShared);
    if (!buffer) {
        std::cerr << "Error: Failed to create buffer for " << bufferName << "!" << std::endl;
        exit(1);
    }
    
    void* contents = buffer->contents();
    if (!contents) {
        std::cerr << "Error: Failed to get contents for " << bufferName << " buffer!" << std::endl;
        exit(1);
    }
    
    std::memcpy(contents, data.data(), byteSize);
    return buffer;
};

template<typename T, typename BufferType>
void metal_print(BufferType* buffer) {
    T* data_buffer = static_cast<T*>(buffer->contents());
    std::cout << "First few input values: " << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << ", Buffer[" << i << "]: " << data_buffer[i] << std::endl;
    }
}

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
        } else {
            std::cout << "Successfully loaded Metal library" << std::endl;
        }
    }
};

template<typename T>
Tensor<T> MetalBackend<T>::ewise_add(Tensor<T>& e1, Tensor<T>& e2) {
    if (e1.shape != e2.shape) {
        throw std::invalid_argument("Tensors must have same shapes for ewise operations");
    }

    MTL::Buffer* buffer_e1 = copyToBuffer<T>(device, e1.data, "e1");
    MTL::Buffer* buffer_e2 = copyToBuffer<T>(device, e2.data, "e2");
    MTL::Buffer* buffer_res = copyToBuffer<T>(device, e2.data, "result");
    metal_print<T>(buffer_e1);
    metal_print<T>(buffer_e2);

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
    encoder->setBuffer(buffer_e2, 1, 0);
    encoder->setBuffer(buffer_res, 2, 0);

    MTL::Size gridSize(e1.data.size(), 1, 1);
    MTL::Size threadgroupSize(64, 1, 1);
    encoder->dispatchThreads(gridSize, threadgroupSize);

    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    metal_print<T>(buffer_res);
    std::vector<T> res(static_cast<T*>(buffer_res->contents()), 
                      static_cast<T*>(buffer_res->contents()) + e1.data.size());

    // Cleanup
    buffer_e1->release();
    buffer_e2->release();
    buffer_res->release();
    pipelineState->release();

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
