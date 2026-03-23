#pragma once

#include <pybind11/pybind11.h>
#include <vector>

// dont declare MTL, just mention it exists
namespace MTL {
class Buffer;
}

template <typename T> struct MetalTensor {
  // data storage
  std::vector<T> cpu_data;
  MTL::Buffer *gpu_buffer = nullptr;
  bool cpu_valid = true;
  bool gpu_valid = false;
  // data structure
  std::vector<size_t> shape;
  std::vector<size_t> stride;
  size_t offset = 0;
  size_t total_size = 0;
  static int tensor_count;

  // Constructor and destructor
  MetalTensor() { tensor_count++; }
  ~MetalTensor();

  std::vector<T> get_data() const;

  void initialize(const std::vector<T> &data, const std::vector<size_t> &shape);
  static MetalTensor<T> create(const std::vector<T> &data,
                               const std::vector<size_t> &shape);
  MetalTensor<T> randn(const std::vector<int> &size, int mean, int std);
  MetalTensor<T> fill(const std::vector<int> &size, T val);
  void compact();
  bool is_contiguous() const;
  void set_metadata(const std::vector<size_t> &shape,
                    const std::vector<size_t> &stride, size_t offset);
  size_t mult_dim_to_flat_index(const std::vector<size_t> &dimension) const;
  std::vector<size_t> flat_index_to_mult_dim(const size_t index) const;
  void reshape(const std::vector<size_t> &new_shape);
  void swap(const size_t axis1, const size_t axis2);
  int get_tensor_count();
  MetalTensor<T> max(const std::vector<size_t> &axes, const bool keep_dims);
  void print() const;
  T get_item(const std::vector<size_t> &index) const;
  void set_item(std::vector<size_t> &indices, T value);

private:
  static size_t m_calculate_size(const std::vector<size_t> &shape);
  static std::vector<size_t>
  m_calculate_stride(const std::vector<size_t> &shape);
};

template <typename T>
int MetalTensor<T>::tensor_count = 0;

template <typename T>
void bind_metal_tensor(pybind11::module &m, const std::string &class_name);

#include "tensor.tpp"
