#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>

namespace py = pybind11;

template<typename T>
struct Tensor {
    std::vector<T> data;
    std::vector<size_t> shape;
    std::vector<size_t> stride;
    size_t offset;

  static Tensor<T> initialize(const std::vector<T>& data, const std::vector<size_t>& shape){
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

private:
    static size_t calculate_size(const std::vector<size_t>& shape) {
        size_t r_size = 1;
        for (size_t s : shape) { 
            r_size *= s;
        }
        return r_size;
    }

    static std::vector<size_t> calculate_stride(const std::vector<size_t>& shape) {
        std::vector<size_t> stride(shape.size());
        size_t product = 1;
        for(int i = shape.size() - 1; i >= 0; --i) {
            stride[i] = product;
            product *= shape[i];
        }
        return stride;
    }

public:
    size_t mult_dim_to_flat_index(const std::vector<size_t>& dimension) const {
        size_t result = offset;
        for (int i = 0; i < shape.size(); i++) {
            result += dimension[i] * stride[i];
        }
        return result;
    }


    std::vector<size_t> flat_index_to_mult_dim(const size_t index) const {
        std::vector<size_t> result(shape.size());
        size_t current = index;
        for(int i = shape.size() - 1; i >= 0; --i) {
            result[i] = current % shape[i];
            current /= shape[i];
        }
        return result;
    }
};



template<typename T>
class CPUBackend {
public:
    CPUBackend() = default;

    Tensor<T> ewise_add(Tensor<T> e1, Tensor<T> e2) {
        if (e1.shape != e2.shape) {
            throw std::invalid_argument("Tensors must have same shapes for ewise operations");
        }
        std::vector<T> result_data(e1.data.size());
        for(int i = 0; i < e1.data.size(); i++) {
            std::vector<size_t> multi_dim = e1.flat_index_to_mult_dim(i);
            size_t e1_index = e1.mult_dim_to_flat_index(multi_dim);
            size_t e2_index = e2.mult_dim_to_flat_index(multi_dim);
            result_data[i] = e1.data[e1_index] + e2.data[e2_index];
        }
        return Tensor<T>::initialize(result_data, e1.shape);
    }

    Tensor<T> ewise_mul(Tensor<T> e1, Tensor<T> e2) {
        if (e1.shape != e2.shape) {
            throw std::invalid_argument("Tensors must have same shapes for ewise operations");
        }
        std::vector<T> result_data(e1.data.size());
        for(int i = 0; i < e1.data.size(); i++) {
            std::vector<size_t> multi_dim = e1.flat_index_to_mult_dim(i);
            size_t e1_index = e1.mult_dim_to_flat_index(multi_dim);
            size_t e2_index = e2.mult_dim_to_flat_index(multi_dim);
            result_data[i] = e1.data[e1_index] * e2.data[e2_index];
        }
        return Tensor<T>::initialize(result_data, e1.shape);
    }

    // // Make our array contiguous. Many matrix operation are implemented by manipulating
    // // shape, stride, and offset. However, some operations requrie our matrix to be compact..
    // void compact(std::vector<int> input, std::vector<int32_t> shape, std::vector<int32_t> stride, size_t offset) {}
};

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
        .def("mult_dim_to_flat_index", &Tensor<T>::mult_dim_to_flat_index);
}

template <typename T>
void bind_operations(pybind11::module& m, const std::string& class_name) {
    py::class_<CPUBackend<T>>(m, class_name.c_str())
        .def(py::init<>())
        .def("ewise_add", &CPUBackend<T>::ewise_add)
        .def("ewise_mul", &CPUBackend<T>::ewise_mul);
}


PYBIND11_MODULE(cpu_backend, m) {
    bind_operations<int32_t>(m, "CPUBackendInt");
    bind_operations<int64_t>(m, "CPUBackendLong");
    bind_operations<float>(m, "CPUBackendFloat");
    bind_operations<double>(m, "CPUBackendDouble");

    bind_tensor<int32_t>(m, "IntTensor");
    bind_tensor<int64_t>(m, "LongTensor");
    bind_tensor<float>(m, "FloatTensor");
    bind_tensor<double>(m, "DoubleTensor");
}
