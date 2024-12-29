#ifndef TENSOR_H
#define TENSOR_H

#include <pybind11/pybind11.h>
#include <vector>

template<typename T>
struct Tensor {
    std::vector<T> data;
    std::vector<size_t> shape;
    std::vector<size_t> stride;
    size_t offset;

    static Tensor<T> initialize(const std::vector<T>& data, const std::vector<size_t>& shape);
    void compact();
    size_t mult_dim_to_flat_index(const std::vector<size_t>& dimension) const;
    std::vector<size_t> flat_index_to_mult_dim(const size_t index) const;

private:
    static size_t calculate_size(const std::vector<size_t>& shape);
    static std::vector<size_t> calculate_stride(const std::vector<size_t>& shape);
};

template <typename T>
void bind_tensor(pybind11::module& m, const std::string& class_name);

#include "tensor.tpp" // Include implementation here

#endif // TENSOR_H
