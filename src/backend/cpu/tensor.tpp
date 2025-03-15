#pragma once

#include "tensor.h"
#include <random>

template<typename T>
Tensor<T> Tensor<T>::initialize(const std::vector<T>& data, const std::vector<size_t>& shape) {
    if (data.size() != calculate_size(shape)) {
        throw std::invalid_argument("Data size does not match shape dimensions.");
    }
    Tensor<T> tensor;
    tensor.data = data;
    tensor.shape = shape;
    tensor.stride = calculate_stride(shape);
    tensor.offset = 0;
    return tensor;
}

template<typename T>
std::vector<T> Tensor<T>::randn(const std::vector<int>& size, int mean, int std) {
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
void Tensor<T>::print() const {
    std::cout << "Tensor Information:\n";
    std::cout << "Shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
    }
    std::cout << "]\n";

    std::cout << "Stride: [";
    for (size_t i = 0; i < stride.size(); ++i) {
        std::cout << stride[i] << (i < stride.size() - 1 ? ", " : "");
    }
    std::cout << "]\n";

    std::cout << "Data: [";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << data[i] << (i < data.size() - 1 ? ", " : "");
    }
    std::cout << "]\n";
}

template<typename T>
std::vector<T> Tensor<T>::create(const std::vector<int>& size, T val) {
    size_t n = std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>());
    std::vector<T> data(n, val);
    return data;
}

template <typename T>
void Tensor<T>::reshape(const std::vector<size_t>& new_shape) {
    if (calculate_size(new_shape) != data.size()) {
        throw std::invalid_argument("New shape must have the same number of elements.");
    }
    shape = new_shape;
    stride = calculate_stride(new_shape);
}

// Make our array contiguous. Many matrix operation are implemented by manipulating
// shape, stride, and offset. However, some operations require our matrix to be compact..
template<typename T>
void Tensor<T>::compact() {
    size_t num_elements = 1;
    for (size_t elem: shape) {
        num_elements *= elem;
    }
    std::vector<T> new_data(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        std::vector<size_t> multi_dim = flat_index_to_mult_dim(i);
        size_t source_index = mult_dim_to_flat_index(multi_dim);
        new_data[i] = data[source_index];
    }
    data = new_data;
    stride = calculate_stride(shape);
    offset = 0;
}

template<typename T>
size_t Tensor<T>::calculate_size(const std::vector<size_t>& shape) {
    size_t r_size = 1;
    for (size_t s : shape) {
        r_size *= s;
    }
    return r_size;
}

template<typename T>
std::vector<size_t> Tensor<T>::calculate_stride(const std::vector<size_t>& shape) {
    std::vector<size_t> stride(shape.size());
    size_t product = 1;
    for(int i = shape.size() - 1; i >= 0; --i) {
        stride[i] = product;
        product *= shape[i];
    }
    return stride;
}

template<typename T>
size_t Tensor<T>::mult_dim_to_flat_index(const std::vector<size_t>& dimension) const {
    size_t result = offset;
    for (int i = 0; i < shape.size(); i++) {
        result += dimension[i] * stride[i];
    }
    return result;
}

template<typename T>
std::vector<size_t> Tensor<T>::flat_index_to_mult_dim(const size_t index) const {
    std::vector<size_t> result(shape.size());
    size_t current = index;
    for(int i = shape.size() - 1; i >= 0; --i) {
        result[i] = current % shape[i];
        current /= shape[i];
    }
    return result;
}

template<typename T>
void Tensor<T>::swap(const size_t axis1, const size_t axis2) {
    this->print();
    std::swap(this->shape[axis1], this->shape[axis2]);
    std::swap(this->stride[axis1], this->stride[axis2]);
}

template <typename T>
void bind_tensor(pybind11::module& m, const std::string& class_name) {
    pybind11::class_<Tensor<T>>(m, class_name.c_str())
        .def(pybind11::init<>())
        .def_readwrite("data", &Tensor<T>::data)
        .def_readwrite("shape", &Tensor<T>::shape)
        .def_readwrite("stride", &Tensor<T>::stride)
        .def_readwrite("offset", &Tensor<T>::offset)
        .def_static("initialize", &Tensor<T>::initialize, "Initialize a Tensor",
                    pybind11::arg("data"), pybind11::arg("shape"))
        .def("randn", &Tensor<T>::randn, "Generate a random Tensor")
        .def("create", &Tensor<T>::create)
        .def("print", &Tensor<T>::print)
        .def("compact", &Tensor<T>::compact, "Compact a Tensor")
        .def("reshape", &Tensor<T>::reshape, "Reshape a Tensor")
        .def("mult_dim_to_flat_index", &Tensor<T>::mult_dim_to_flat_index)
        .def("swap", &Tensor<T>::swap, "swap shape and stride of a tensor",
            pybind11::arg("axis1"), pybind11::arg("axis2"));
}
