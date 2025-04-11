#pragma once

#include <pybind11/pybind11.h>
#include <vector>

template<typename T>
struct Tensor {
    std::shared_ptr<std::vector<T>> data;
    std::vector<size_t> shape;
    std::vector<size_t> stride;
    size_t offset;
    size_t total_size;
    static int tensor_count;

    Tensor() {
        tensor_count++;
//        std::cout << "Default constructor: " << Tensor<T>::tensor_count << std::endl;
    }

    ~Tensor() {
        tensor_count--;
//        std::cout << "Destructor: " << Tensor<T>::tensor_count << std::endl;
    }

    Tensor(const std::vector<T>& data, const std::vector<size_t>& shape) {
        this->data = std::make_shared<std::vector<T>>(data);
        this->shape = shape;
        this->stride = calculate_stride(shape);
        this->offset = 0;
        this->total_size = calculate_size(shape);
        Tensor<T>::tensor_count++;
    }

    Tensor(const std::vector<T>& data, const std::vector<size_t>& shape, const std::vector<size_t>& stride, size_t offset) {
        this->data = std::make_shared<std::vector<T>>(data);
        this->shape = shape;
        this->stride = stride;
        this->offset = offset;
        this->total_size = calculate_size(shape);
        Tensor<T>::tensor_count++;
    }

    std::vector<T> get_data() const;

    void initialize(const std::vector<T>& data, const std::vector<size_t>& shape);

    static Tensor<T> create(const std::vector<T>& data, const std::vector<size_t>& shape);

    static Tensor<T> create(const std::vector<T>& data, const std::vector<size_t>& shape,
        const std::vector<size_t>& stride, const size_t offset);

    static Tensor<T> create_view(const std::shared_ptr<std::vector<T>>& data, const std::vector<size_t>& shape,
        const std::vector<size_t>& stride, size_t offset);

    Tensor<T> randn(const std::vector<size_t>& size, int mean, int std);

    Tensor<T> fill(const std::vector<size_t>& size, T val);

    void compact();

    size_t mult_dim_to_flat_index(const std::vector<size_t>& dimension) const;

    std::vector<size_t> flat_index_to_mult_dim(const size_t index) const;

    void reshape(const std::vector<size_t>& new_shape);

    void swap(const size_t axis1, const size_t axis2);

    int get_tensor_count();

    Tensor<T> max(std::vector<size_t>& axes);

    void print() const;

private:
    static size_t calculate_size(const std::vector<size_t>& shape);

    static std::vector<size_t> calculate_stride(const std::vector<size_t>& shape);
};

template <typename T>
int Tensor<T>::tensor_count = 0;

template <typename T>
void bind_tensor(pybind11::module& m, const std::string& class_name);

#include "tensor.tpp"
