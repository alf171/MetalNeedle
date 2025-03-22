#pragma once

#include <pybind11/pybind11.h>
#include <vector>

// dont declare MTL, just mention it exists
namespace MTL {
    class Buffer;
}

template<typename T>
struct MetalTensor {
    // data storage
    std::vector<T> cpu_data;
    MTL::Buffer* gpu_buffer = nullptr;
    bool cpu_valid = true;
    bool gpu_valid = false;
    // data structure
    std::vector<size_t> shape;
    std::vector<size_t> stride;
    size_t offset = 0;

    // Constructor and destructor
    MetalTensor() = default;
    ~MetalTensor();

    static MetalTensor<T> initialize(const std::vector<T>& data, const std::vector<size_t>& shape);
    std::vector<T> randn(const std::vector<int>& size, int mean, int std);
    std::vector<T> create(const std::vector<int>& size, T val);
    void compact();
    size_t mult_dim_to_flat_index(const std::vector<size_t>& dimension) const;
    std::vector<size_t> flat_index_to_mult_dim(const size_t index) const;
    void reshape(const std::vector<size_t>& new_shape);
    void print() const;

private:
    static size_t calculate_size(const std::vector<size_t>& shape);
    static std::vector<size_t> calculate_stride(const std::vector<size_t>& shape);
};

template <typename T>
void bind_metal_tensor(pybind11::module& m, const std::string& class_name);

#include "tensor.tpp"
