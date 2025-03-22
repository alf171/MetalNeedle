#pragma once

#include "tensor.h"
#include <random>

#include <Metal/Metal.hpp>

// folowing RAII principles and memory shared around using make_unique
template<typename T>
MetalTensor<T> MetalTensor<T>::initialize(const std::vector<T>& data, const std::vector<size_t>& shape) {
    if (data.size() != calculate_size(shape)) {
        throw std::invalid_argument("Data size does not match shape dimensions.");
    }
    MetalTensor<T> tensor;
    tensor.cpu_data = data;
    tensor.shape = shape;
    tensor.stride = calculate_stride(shape);
    tensor.offset = 0;
    tensor.gpu_buffer = nullptr;
    tensor.cpu_valid = true;
    tensor.gpu_valid = false;
    return tensor;
}

template<typename T>
MetalTensor<T>::~MetalTensor() {
    if (gpu_buffer) {
        gpu_buffer->release();
        gpu_buffer = nullptr;
    }
    // (cpu_data, shape, stride, offset) will be automatically cleaned up
}

template<typename T>
std::vector<T> MetalTensor<T>::randn(const std::vector<int>& size, int mean, int std) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<double> dist(mean, std);

    size_t n = std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>());
    std::vector<T> data;
    data.reserve(n);

    for(size_t i = 0; i < n; ++i) {
        data.push_back(static_cast<T>(dist(gen)));
    }

    return data;
}

template<typename T>
void MetalTensor<T>::print() const {
    std::cout << "Tensor Information:\n";
    std::cout << "Shape: [";
    for (size_t i = 0; i < this->cpu_data.size(); ++i) {
        std::cout << this->shape[i] << (i < this->shape.size() - 1 ? ", " : "");
    }
    std::cout << "]\n";

    std::cout << "Stride: [";
    for (size_t i = 0; i < this->stride.size(); ++i) {
        std::cout << this->stride[i] << (i < this->stride.size() - 1 ? ", " : "");
    }
    std::cout << "]\n";

    std::cout << "Data: [";
    for (size_t i = 0; i < this->cpu_data.size(); ++i) {
        std::cout << this->cpu_data[i] << (i < this->cpu_data.size() - 1 ? ", " : "");
    }
    std::cout << "]\n";
}

template<typename T>
std::vector<T> MetalTensor<T>::create(const std::vector<int>& size, T val) {
    size_t n = std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>());
    std::vector<T> data(n, val);
    return data;
}

template <typename T>
void MetalTensor<T>::reshape(const std::vector<size_t>& new_shape) {
    if (calculate_size(new_shape) != this->cpu_data.size()) {
        throw std::invalid_argument("New shape must have the same number of elements.");
    }
    this->shape = new_shape;
    this->stride = calculate_stride(new_shape);
}

// Make our array contiguous. Many matrix operation are implemented by manipulating
// shape, stride, and offset. However, some operations require our matrix to be compact..
template<typename T>
void MetalTensor<T>::compact() {
    size_t num_elements = 1;
    for (size_t elem : this->shape) {
        num_elements *= elem;
    }
    std::vector<T> new_data(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        std::vector<size_t> multi_dim = flat_index_to_mult_dim(i);
        size_t source_index = mult_dim_to_flat_index(multi_dim);
        new_data[i] = this->cpu_data[source_index];
    }
    this->cpu_data = new_data;
    this->stride = calculate_stride(this->shape);
    this->offset = 0;
}

template<typename T>
size_t MetalTensor<T>::calculate_size(const std::vector<size_t>& shape) {
    size_t r_size = 1;
    for (size_t s : shape) {
        r_size *= s;
    }
    return r_size;
}

template<typename T>
std::vector<size_t> MetalTensor<T>::calculate_stride(const std::vector<size_t>& shape) {
    std::vector<size_t> stride(shape.size());
    size_t product = 1;
    for(int i = shape.size() - 1; i >= 0; --i) {
        stride[i] = product;
        product *= shape[i];
    }
    return stride;
}

template<typename T>
size_t MetalTensor<T>::mult_dim_to_flat_index(const std::vector<size_t>& dimension) const {
    size_t result = this->offset;
    for (int i = 0; i < this->shape.size(); i++) {
        result += dimension[i] * this->stride[i];
    }
    return result;
}

template<typename T>
std::vector<size_t> MetalTensor<T>::flat_index_to_mult_dim(const size_t index) const {
    std::vector<size_t> result(this->shape.size());
    size_t current = index;
    for(int i = shape.size() - 1; i >= 0; --i) {
        result[i] = current % this->shape[i];
        current /= this->shape[i];
    }
    return result;
}

template <typename T>
void bind_metal_tensor(pybind11::module& m, const std::string& class_name) {
    pybind11::class_<MetalTensor<T>>(m, class_name.c_str())
        .def(pybind11::init<>())
        .def_readwrite("data", &MetalTensor<T>::cpu_data)
        .def_readwrite("shape", &MetalTensor<T>::shape)
        .def_readwrite("stride", &MetalTensor<T>::stride)
        .def_readwrite("offset", &MetalTensor<T>::offset)
        .def_static("initialize", &MetalTensor<T>::initialize, "Initialize a Metal Tensor",
                    pybind11::arg("data"), pybind11::arg("shape"))
        .def("randn", &MetalTensor<T>::randn, "Generate a random Tensor")
        .def("create", &MetalTensor<T>::create)
        .def("print", &MetalTensor<T>::print)
        .def("compact", &MetalTensor<T>::compact, "Compact a Tensor")
        .def("reshape", &MetalTensor<T>::reshape, "Reshape a Tensor")
        .def("mult_dim_to_flat_index", &MetalTensor<T>::mult_dim_to_flat_index);
}
