#pragma once

#include "tensor.h"
#include <random>

template<typename T>
void Tensor<T>::initialize(const std::vector<T>& data, const std::vector<size_t>& shape) {
    size_t total_size = calculate_size(shape);
    if (data.size() != total_size) {
        throw std::invalid_argument("Data size does not match shape dimensions.");
    }
    this->data = std::make_shared<std::vector<T>>(data);
    this->shape = shape;
    this->stride = calculate_stride(shape);
    this->offset = 0;
    this->total_size = total_size;
}

template<typename T>
Tensor<T> Tensor<T>::create(const std::vector<T>& data, const std::vector<size_t>& shape) {
    return Tensor<T>(data, shape);
}

template<typename T>
Tensor<T> Tensor<T>::create(const std::vector<T>& data, const std::vector<size_t>& shape, const std::vector<size_t>& stride,
        size_t offset) {
    return Tensor<T>(data, shape, stride, offset);
}

template<typename T>
Tensor<T> Tensor<T>::create_view(const std::shared_ptr<std::vector<T>>& shared_data,
                            const std::vector<size_t>& shape,
                            const std::vector<size_t>& stride,
                            size_t offset) {
    Tensor<T> tensor;
    tensor.data = shared_data;
    tensor.shape = shape;
    tensor.stride = stride;
    tensor.offset = offset;
    return tensor;
}

template<typename T>
Tensor<T> Tensor<T>::randn(const std::vector<size_t>& size, int mean, int std) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<double> dist(mean, std);

    size_t n = std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>());
    std::vector<T> data;
    data.reserve(n);

    for(size_t i = 0; i < n; ++i) {
        data.push_back(static_cast<T>(dist(gen)));
    }

    return Tensor<T>(data, size);
}

template<typename T>
void Tensor<T>::print() const {
    std::cout << "Tensor Information:\n";
    std::cout << "Shape: [";
    for (size_t i = 0; i < this->shape.size(); ++i) {
        std::cout << this->shape[i] << (i < this->shape.size() - 1 ? ", " : "");
    }
    std::cout << "]\n";

    std::cout << "Stride: [";
    for (size_t i = 0; i < this->stride.size(); ++i) {
        std::cout << this->stride[i] << (i < this->stride.size() - 1 ? ", " : "");
    }
    std::cout << "]\n";

    std::cout << "Data size: " << this->data->size() << std::endl;
    std::cout << "Offset: " << this->offset << std::endl;
    std::cout << "Total size: " << this->total_size << std::endl;
}

template<typename T>
std::vector<T> Tensor<T>::get_data() const {
    return *this->data;
}

template<typename T>
Tensor<T> Tensor<T>::fill(const std::vector<size_t>& size, T val) {
    size_t n = std::accumulate(size.begin(), size.end(), 1, std::multiplies<int>());
    std::vector<T> data(n, val);
    return Tensor<T>(data, size);
}

template <typename T>
void Tensor<T>::reshape(const std::vector<size_t>& new_shape) {
    size_t new_total_size = calculate_size(new_shape);
    if (calculate_size(new_shape) != this->data->size()) {
        throw std::invalid_argument("New shape must have the same number of elements.");
    }
    this->shape = new_shape;
    this->stride = calculate_stride(new_shape);
    this->total_size = new_total_size;
}

// Make our array contiguous. Many matrix operation are implemented by manipulating
// shape, stride, and offset. However, some operations require our matrix to be compact..
template<typename T>
void Tensor<T>::compact() {
    std::vector<T> new_data(this->total_size);
    for (size_t i = 0; i < num_elements; ++i) {
        std::vector<size_t> multi_dim = flat_index_to_mult_dim(i);
        size_t source_index = mult_dim_to_flat_index(multi_dim);
        new_data[i] = this->data->at(source_index);
    }
    this->data = std::make_shared<std::vector<T>>(new_data);
    this->stride = calculate_stride(shape);
    this->offset = 0;
}

template<typename T>
size_t Tensor<T>::calculate_size(const std::vector<size_t>& input_shape) {
    size_t r_size = 1;
    for (size_t s : input_shape) {
        r_size *= s;
    }
    return r_size;
}

template<typename T>
std::vector<size_t> Tensor<T>::calculate_stride(const std::vector<size_t>& input_shape) {
    std::vector<size_t> stride(input_shape.size());
    size_t product = 1;
    for(int i = input_shape.size() - 1; i >= 0; --i) {
        stride[i] = product;
        product *= input_shape[i];
    }
    return stride;
}

template<typename T>
size_t Tensor<T>::mult_dim_to_flat_index(const std::vector<size_t>& dimension) const {
    if (dimension.size() != this->shape.size()) {
        throw std::invalid_argument("[mult_dim_to_flat_index] expected dim of " + std::to_string(this->shape.size()) + " but got " + std::to_string(dimension.size()));
    }
    size_t result = this->offset;
    for (int i = 0; i < this->shape.size(); i++) {
        result += dimension[i] * this->stride[i];
    }
    return result;
}

template<typename T>
std::vector<size_t> Tensor<T>::flat_index_to_mult_dim(const size_t index) const {
    std::vector<size_t> result(this->shape.size());
    size_t current = index;
    for(int i = this->shape.size() - 1; i >= 0; --i) {
        result[i] = current % this->shape[i];
        current /= this->shape[i];
    }
    return result;
}

template<typename T>
void Tensor<T>::swap(const size_t axis1, const size_t axis2) {
    std::swap(this->shape[axis1], this->shape[axis2]);
    std::swap(this->stride[axis1], this->stride[axis2]);
}

template<typename T>
int Tensor<T>::get_tensor_count() {
    return this->tensor_count;
}

template<typename T>
Tensor<T> Tensor<T>::max(const std::vector<size_t>& axes, const bool keep_dims) {
    std::vector<bool> reduce_dim(this->shape.size(), true);
    for (size_t axis : axes) {
        if (axis >= reduce_dim.size()) {
            throw std::out_of_range("Axis out of bounds");
        }
        reduce_dim[axis] = false;
    }

    std::vector<size_t> shape;
    for (size_t dim = 0; dim < this->shape.size(); dim++) {
        if(reduce_dim[dim]) {
            shape.push_back(this->shape[dim]);
        } else if (keep_dims) {
            shape.push_back(1);
        }
    }

    if (shape.empty()) {
        shape = {1};
    }

    std::vector<T> data(this->calculate_size(this->shape), std::numeric_limits<T>::lowest());
    Tensor<T> result_tensor = Tensor<T>(data, shape, calculate_stride(shape), 0);

    for(size_t i = 0; i < this->total_size; i++) {
        std::vector<size_t> multi_dim = this->flat_index_to_mult_dim(i);
        std::vector<size_t> result_indices;
        for (size_t dim = 0; dim < multi_dim.size(); dim++) {
            // if we aren't reducing across dim
            if(reduce_dim[dim]) {
                result_indices.push_back(multi_dim[dim]);
            } else if (keep_dims) {
                result_indices.push_back(0);
            }
        }

        size_t index = result_tensor.mult_dim_to_flat_index(result_indices);
        if (this->data->at(i) > result_tensor.data->at(index)) {
            result_tensor.data->at(index) = this->data->at(i);
        }
    }
    return result_tensor;
}

template <typename T>
void bind_tensor(pybind11::module& m, const std::string& class_name) {
    // TODO: should be just read
    pybind11::class_<Tensor<T>>(m, class_name.c_str())
        .def(pybind11::init<>())
        .def("data", &Tensor<T>::get_data)
        .def_readwrite("shape", &Tensor<T>::shape)
        .def_readwrite("stride", &Tensor<T>::stride)
        .def_readwrite("offset", &Tensor<T>::offset)
        .def("initialize", &Tensor<T>::initialize, "Initialize a Tensor",
            pybind11::arg("data"), pybind11::arg("shape"))
        .def_static("create", [](const std::vector<T>& data, const std::vector<size_t>& shape) {
              return Tensor<T>::create(data, shape);
           },
           "Factory method to make a tensor with data and shape",
           pybind11::arg("data"), pybind11::arg("shape"))
        .def("randn", &Tensor<T>::randn, "Generate a random Tensor")
        .def("fill", &Tensor<T>::fill)
        .def("print", &Tensor<T>::print)
        .def("compact", &Tensor<T>::compact, "Compact a Tensor")
        .def("reshape", &Tensor<T>::reshape, "Reshape a Tensor")
        .def("mult_dim_to_flat_index", &Tensor<T>::mult_dim_to_flat_index)
        .def("swap", &Tensor<T>::swap, "swap shape and stride of a tensor",
            pybind11::arg("axis1"), pybind11::arg("axis2"))
        .def("max", &Tensor<T>::max, "get the max value of a tensor")
        .def("get_tensor_count", &Tensor<T>::get_tensor_count);
}
