#include <algorithm>
// for logging
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;
// this number might have to be tuned to match width of asm instruction
#define TILE static_cast<size_t>(1)

template<typename T>
struct Tensor {
    std::vector<T> data;
    std::vector<size_t> shape;
    std::vector<size_t> stride;
    size_t offset;

  static Tensor<T> initialize(const std::vector<T>& data, const std::vector<size_t>& shape) {
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

    // Make our array contiguous. Many matrix operation are implemented by manipulating
    // shape, stride, and offset. However, some operations require our matrix to be compact..
    void compact() {
        size_t num_elements = 1;
        for (size_t elem : shape) {
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

    /**
     * description: result = (e1 + e2) \forall e1,e2 \in Tensor1, Tensor2
     * input: e1: Tensor, e2: Tensor
     * output: result: Tensor
    **/ 
    Tensor<T> ewise_add(Tensor<T>& e1, Tensor<T>& e2) {
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


    /**
     * description: result = (e1 - e2) \forall e1,e2 \in Tensor1, Tensor2
     * input: e1: Tensor, e2: Tensor
     * output: result: Tensor
    **/ 
    Tensor<T> ewise_sub(Tensor<T>& e1, Tensor<T>& e2) {
        if (e1.shape != e2.shape) {
            throw std::invalid_argument("Tensors must have same shapes for ewise operations");
        }
        std::vector<T> result_data(e1.data.size());
        for(int i = 0; i < e1.data.size(); i++) {
            std::vector<size_t> multi_dim = e1.flat_index_to_mult_dim(i);
            size_t e1_index = e1.mult_dim_to_flat_index(multi_dim);
            size_t e2_index = e2.mult_dim_to_flat_index(multi_dim);
            result_data[i] = e1.data[e1_index] - e2.data[e2_index];
        }
        return Tensor<T>::initialize(result_data, e1.shape);
    }

    /**
     * description: result = (e1 ** pow) \forall e1 \in Tensor1
     * input: Tensor1: Tensor, pow: float
     * output: Tensor: Tensor
    **/
    Tensor<T> ewise_exp(Tensor<T>& e1, float v1) {
        std::vector<T> result_data(e1.data.size());
        for(int i = 0; i < e1.data.size(); i++) {
            std::vector<size_t> multi_dim = e1.flat_index_to_mult_dim(i);
            size_t e1_index = e1.mult_dim_to_flat_index(multi_dim);
            result_data[i] = pow(e1.data[e1_index], v1);
        }
        return Tensor<T>::initialize(result_data, e1.shape);
    }

    /**
     * description: result = \sum_{j} (Tensor1_ij * Tensor2_jk) \forall i,k 
     * input: Tensor1: Tensor, Tensor2: Tensor
     * output: result: Tensor
    **/
    Tensor<T> ewise_mul(Tensor<T>& e1, Tensor<T>& e2) {
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

    Tensor<T> tiled_mat_mul(Tensor<T>& e1, Tensor<T>& e2) {
        size_t e1_last_dim = e1.shape.size() - 1;
        size_t e2_last_dim = e2.shape.size() - 1;
        // this could be moved into data manipulation loop
        if (e1.shape[e1_last_dim] != e2.shape[0]) {
            throw std::invalid_argument("matmul shapes are not congruent");
        }

        std::cout << "Matrix 1 shape: (" << e1.shape[0] << ", " << e1.shape[1] << ")" << std::endl;
        std::cout << "Matrix 2 shape: (" << e2.shape[0] << ", " << e2.shape[1] << ")" << std::endl;

        // make matrices compact to have better caching properties
        e1.compact();
        e2.compact();
            
        // (m,n) @ (n,p) => (m,p)
        size_t e1_rows = e1.shape[0];
        size_t e2_cols = e2.shape[1];
        size_t e1_cols = e1.shape[e1_last_dim];

        std::vector<size_t> result_shape = {e1_rows, e2_cols};
        std::vector<T> result_data(e1_rows * e2_cols, 0);

        // add another for loop in order to support dim > 2
        for(size_t block_x = 0; block_x < e1.shape[0]; block_x += TILE) {
            for(size_t block_y = 0; block_y < e2.shape[e2_last_dim]; block_y += TILE) {
                tile_compute(e1.data, e2.data, result_data, block_x, block_y, e1_cols, e2_cols, e1_rows);
            }
        }

        Tensor<T> result_tensor = Tensor<T>::initialize(result_data, result_shape);
        return result_tensor;
    }

    /**
     * description: result = e1 + scalar for all e1 in tensor
       * this method is not destructive i.e. it uses the tensor passed in
       * unlike other ops which generate a new one
       * TODO: this is a nasty habit so should fix
     * input: tensor: Tensor, scalar: T
     * output: result: Tensor
    **/
    Tensor<T> scalar_add(Tensor<T>& tensor, T scalar) {
        std::vector<T> result_data(tensor.data.size());
        for(int i = 0; i < tensor.data.size(); i++) {
            result_data[i] = tensor.data[i] + scalar;
        }
        return Tensor<T>::initialize(result_data, tensor.shape);
    }

    /**
     * description: result = tensor - scalar
     * input: tensor: Tensor, scalar: T
     * output: result: Tensor
    **/
    Tensor<T> scalar_sub(Tensor<T>& tensor, T scalar) {
        std::vector<T> result_data(tensor.data.size());
        for(int i = 0; i < tensor.data.size(); i++) {
            result_data[i] = tensor.data[i] - scalar;
        }
        return Tensor<T>::initialize(result_data, tensor.shape);
    }

    /**
     * description: result = tensor * scalar
     * input: tensor: Tensor, scalar: T
     * output: result: Tensor
    **/
    Tensor<T> scalar_mul(Tensor<T>& tensor, T scalar) {
        std::vector<T> result_data(tensor.data.size());
        for(int i = 0; i < tensor.data.size(); i++) {
            result_data[i] = tensor.data[i] * scalar;
        }
        return Tensor<T>::initialize(result_data, tensor.shape);
    }

    /**
     * description: result = tensor / scalar
     * input: tensor: Tensor, scalar: T
     * output: result: Tensor
    **/
    Tensor<T> scalar_div(Tensor<T>& tensor, T scalar) {
        std::vector<T> result_data(tensor.data.size());
        for(int i = 0; i < tensor.data.size(); i++) {
            result_data[i] = tensor.data[i] / scalar;
        }
        return Tensor<T>::initialize(result_data, tensor.shape);
    }

private:
    void tile_compute(std::vector<T>& e1, std::vector<T>& e2, std::vector<T>& res,
                      size_t block_x, size_t block_y, size_t e1_cols, size_t e2_cols, size_t e1_rows) {

        // adjust for when SIZE % TILE != 0
        size_t tile_height = std::min(TILE, e1_rows - block_x);
        size_t tile_width = std::min(TILE, e2_cols - block_y);

        for (size_t i = 0; i < tile_height; i++) {
            for (size_t j = 0; j < tile_width; j++) {
                T tmp_sum = 0;
                for (size_t k = 0; k < e1_cols; k++) {
                    size_t index_e1 = (block_x + i) * e1_cols + k;
                    size_t index_e2 = k * e2_cols + (block_y + j);
                    tmp_sum += e1[index_e1] * e2[index_e2];
                }
                size_t index_res = (block_x + i) * e2_cols + (block_y + j);
                res[index_res] += tmp_sum;
            }
        }
    }
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
//        .def("compact", &Tensor<T>::compact, "Compact a Tensor")
        .def("mult_dim_to_flat_index", &Tensor<T>::mult_dim_to_flat_index);
}

template <typename T>
void bind_operations(pybind11::module& m, const std::string& class_name) {
    py::class_<CPUBackend<T>>(m, class_name.c_str())
        .def(py::init<>())
        .def("ewise_add", &CPUBackend<T>::ewise_add)
        .def("ewise_sub", &CPUBackend<T>::ewise_sub)
        .def("ewise_exp", &CPUBackend<T>::ewise_exp)
        .def("ewise_mul", &CPUBackend<T>::ewise_mul)
        .def("mat_mul", &CPUBackend<T>::tiled_mat_mul)
        .def("scalar_add", &CPUBackend<T>::scalar_add)
        .def("scalar_sub", &CPUBackend<T>::scalar_sub)
        .def("scalar_mul", &CPUBackend<T>::scalar_mul)
        .def("scalar_div", &CPUBackend<T>::scalar_div);
}

void bind_cpu(py::module &m) {
    auto cpu = m.def_submodule("cpu");
    // operations
    bind_operations<int32_t>(cpu, "IntOperation");
    bind_operations<int64_t>(cpu, "LongOperation");
    bind_operations<float>(cpu, "FloatOperation");
    bind_operations<double>(cpu, "DoubleOperation");
    // data
    bind_tensor<int32_t>(cpu, "IntTensor");
    bind_tensor<int64_t>(cpu, "LongTensor");
    bind_tensor<float>(cpu, "FloatTensor");
    bind_tensor<double>(cpu, "DoubleTensor");
}
