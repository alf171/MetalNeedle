#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#include <Metal/Metal.hpp>
#include <type_traits>
#include <unistd.h>
#include "backend.h"
#include "../trace.h"

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
    TRACE_SCOPE("metal.constructor");
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
MTL::ComputePipelineState* MetalBackend<T>::make_pipeline(
    const char* function_name) {
    NS::Error* error = nullptr;
    auto name = NS::String::string(function_name, NS::ASCIIStringEncoding);
    MTL::Function* computeFunction = opLibrary->newFunction(name);
    if (computeFunction == nullptr) {
        std::cerr << "Failed to find function '" << function_name
                  << "' in the Metal library" << std::endl;
        exit(1);
    }

    MTL::ComputePipelineState* pipelineState =
        device->newComputePipelineState(computeFunction, &error);
    computeFunction->release();

    if (error) {
        std::cerr << "Failed to create pipeline state: "
                  << error->localizedDescription()->utf8String() << std::endl;
        exit(1);
    }

    return pipelineState;
}

template<typename T>
MTL::Size MetalBackend<T>::make_threadgroup_size(
    MTL::ComputePipelineState* pipelineState, size_t num_elements) {
    NS::UInteger threadGroupSize = pipelineState->maxTotalThreadsPerThreadgroup();
    if (threadGroupSize > num_elements) {
        threadGroupSize = num_elements;
    }
    return MTL::Size::Make(threadGroupSize, 1, 1);
}

template<typename T>
MetalTensor<T> MetalBackend<T>::run_binary_kernel(const char* function_name,
                                                  MetalTensor<T>& e1,
                                                  MetalTensor<T>& e2) {
    if (e1.shape != e2.shape) {
        throw std::invalid_argument("Tensors must have same shapes for ewise operations");
    }

    size_t num_elements = e1.cpu_data.size();
    MTL::Buffer* buffer_e1 = device->newBuffer(num_elements * sizeof(T), MTL::ResourceStorageModeShared);
    MTL::Buffer* buffer_e2 = device->newBuffer(num_elements * sizeof(T), MTL::ResourceStorageModeShared);
    MTL::Buffer* buffer_res = device->newBuffer(num_elements * sizeof(T), MTL::ResourceStorageModeShared);
    std::memcpy(buffer_e1->contents(), e1.cpu_data.data(), num_elements * sizeof(T));
    std::memcpy(buffer_e2->contents(), e2.cpu_data.data(), num_elements * sizeof(T));

    MTL::ComputePipelineState* pipelineState = make_pipeline(function_name);
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipelineState);
    encoder->setBuffer(buffer_e1, 0, 0);
    encoder->setBuffer(buffer_e2, 0, 1);
    encoder->setBuffer(buffer_res, 0, 2);
    encoder->dispatchThreads(MTL::Size::Make(num_elements, 1, 1),
                             make_threadgroup_size(pipelineState, num_elements));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    std::vector<T> res(static_cast<T*>(buffer_res->contents()),
                      static_cast<T*>(buffer_res->contents()) + num_elements);

    buffer_e1->release();
    buffer_e2->release();
    buffer_res->release();
    pipelineState->release();

    return MetalTensor<T>::create(res, e1.shape);
}

template<typename T>
MetalTensor<T> MetalBackend<T>::run_scalar_kernel(const char* function_name,
                                                  MetalTensor<T>& tensor,
                                                  T scalar) {
    size_t num_elements = tensor.cpu_data.size();
    MTL::Buffer* buffer_in = device->newBuffer(num_elements * sizeof(T), MTL::ResourceStorageModeShared);
    MTL::Buffer* buffer_scalar = device->newBuffer(sizeof(T), MTL::ResourceStorageModeShared);
    MTL::Buffer* buffer_res = device->newBuffer(num_elements * sizeof(T), MTL::ResourceStorageModeShared);
    std::memcpy(buffer_in->contents(), tensor.cpu_data.data(), num_elements * sizeof(T));
    std::memcpy(buffer_scalar->contents(), &scalar, sizeof(T));

    MTL::ComputePipelineState* pipelineState = make_pipeline(function_name);
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipelineState);
    encoder->setBuffer(buffer_in, 0, 0);
    encoder->setBuffer(buffer_scalar, 0, 1);
    encoder->setBuffer(buffer_res, 0, 2);
    encoder->dispatchThreads(MTL::Size::Make(num_elements, 1, 1),
                             make_threadgroup_size(pipelineState, num_elements));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    std::vector<T> res(static_cast<T*>(buffer_res->contents()),
                       static_cast<T*>(buffer_res->contents()) + num_elements);

    buffer_in->release();
    buffer_scalar->release();
    buffer_res->release();
    pipelineState->release();

    return MetalTensor<T>::create(res, tensor.shape);
}

template<typename T>
MetalTensor<T> MetalBackend<T>::ewise_add(MetalTensor<T>& e1, MetalTensor<T>& e2) {
    TRACE_SCOPE("metal.ewise_add");
    return run_binary_kernel("metal_ewise_add", e1, e2);
}

template<typename T>
MetalTensor<T> MetalBackend<T>::scalar_mul(MetalTensor<T>& tensor, T scalar) {
    TRACE_SCOPE("metal.scalar_mul");
    return run_scalar_kernel("metal_scalar_mul", tensor, scalar);
}

template<typename T>
MetalTensor<T> MetalBackend<T>::mat_mul(MetalTensor<T>& e1, MetalTensor<T>& e2) {
    TRACE_SCOPE("metal.mat_mul");
    if constexpr (!std::is_same_v<T, float>) {
        throw std::invalid_argument("metal mat_mul is only implemented for float32");
    }

    if (e1.shape.size() != 2 || e2.shape.size() != 2) {
        throw std::invalid_argument("metal mat_mul currently supports only rank-2 tensors");
    }
    if (!e1.is_contiguous() || !e2.is_contiguous()) {
        throw std::invalid_argument("metal mat_mul currently requires contiguous tensors");
    }
    if (e1.shape[1] != e2.shape[0]) {
        throw std::invalid_argument("matmul shapes are not congruent");
    }

    uint32_t rows_a = static_cast<uint32_t>(e1.shape[0]);
    uint32_t shared_dim = static_cast<uint32_t>(e1.shape[1]);
    uint32_t cols_b = static_cast<uint32_t>(e2.shape[1]);
    size_t out_elems = static_cast<size_t>(rows_a) * cols_b;

    MTL::Buffer* buffer_e1 = device->newBuffer(out_elems == 0 ? 1 : e1.cpu_data.size() * sizeof(T),
                                               MTL::ResourceStorageModeShared);
    MTL::Buffer* buffer_e2 = device->newBuffer(out_elems == 0 ? 1 : e2.cpu_data.size() * sizeof(T),
                                               MTL::ResourceStorageModeShared);
    MTL::Buffer* buffer_rows_a = device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* buffer_cols_b = device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* buffer_shared_dim = device->newBuffer(sizeof(uint32_t), MTL::ResourceStorageModeShared);
    MTL::Buffer* buffer_res = device->newBuffer(out_elems * sizeof(T), MTL::ResourceStorageModeShared);

    std::memcpy(buffer_e1->contents(), e1.cpu_data.data(), e1.cpu_data.size() * sizeof(T));
    std::memcpy(buffer_e2->contents(), e2.cpu_data.data(), e2.cpu_data.size() * sizeof(T));
    std::memcpy(buffer_rows_a->contents(), &rows_a, sizeof(uint32_t));
    std::memcpy(buffer_cols_b->contents(), &cols_b, sizeof(uint32_t));
    std::memcpy(buffer_shared_dim->contents(), &shared_dim, sizeof(uint32_t));

    MTL::ComputePipelineState* pipelineState = make_pipeline("metal_mat_mul");
    MTL::CommandBuffer* commandBuffer = commandQueue->commandBuffer();
    MTL::ComputeCommandEncoder* encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipelineState);
    encoder->setBuffer(buffer_e1, 0, 0);
    encoder->setBuffer(buffer_e2, 0, 1);
    encoder->setBuffer(buffer_rows_a, 0, 2);
    encoder->setBuffer(buffer_cols_b, 0, 3);
    encoder->setBuffer(buffer_shared_dim, 0, 4);
    encoder->setBuffer(buffer_res, 0, 5);

    MTL::Size gridSize = MTL::Size::Make(cols_b, rows_a, 1);
    NS::UInteger maxThreads = pipelineState->maxTotalThreadsPerThreadgroup();
    NS::UInteger tg_x = std::min<NS::UInteger>(16, cols_b == 0 ? 1 : cols_b);
    NS::UInteger tg_y = std::max<NS::UInteger>(1, std::min<NS::UInteger>(16, maxThreads / tg_x));
    tg_y = std::min<NS::UInteger>(tg_y, rows_a == 0 ? 1 : rows_a);
    encoder->dispatchThreads(gridSize, MTL::Size::Make(tg_x, tg_y, 1));
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();

    std::vector<T> res(static_cast<T*>(buffer_res->contents()),
                       static_cast<T*>(buffer_res->contents()) + out_elems);

    buffer_e1->release();
    buffer_e2->release();
    buffer_rows_a->release();
    buffer_cols_b->release();
    buffer_shared_dim->release();
    buffer_res->release();
    pipelineState->release();

    return MetalTensor<T>::create(res, {rows_a, cols_b});
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
        .def("ewise_add", &MetalBackend<T>::ewise_add)
        .def("scalar_mul", &MetalBackend<T>::scalar_mul)
        .def("mat_mul", &MetalBackend<T>::mat_mul);
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
