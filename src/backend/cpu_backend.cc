#include <algorithm>
#include <chrono>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>
#include "tensor.h"

namespace py = pybind11;
// this number might have to be tuned to match width of asm instruction
// without parallelism
// TILE = 1 <2048, 2048> @ <2048, 2048> = 24.50
// TILE = 8 <2048, 2048> @ <2048, 2048> = 24.47
// TILE = 64 <2048, 2048> @ <2048, 2048> = 26.35
#define TILE static_cast<size_t>(32)

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
     * description: result = \sum_{j} (Tensor1_ij / Tensor2_jk) \forall i,k
     * input: Tensor1: Tensor, Tensor2: Tensor
     * output: result: Tensor
    **/
    Tensor<T> ewise_div(Tensor<T>& e1, Tensor<T>& e2) {
        if (e1.shape != e2.shape) {
            throw std::invalid_argument("Tensors must have same shapes for ewise operations");
        }
        std::vector<T> result_data(e1.data.size());
        for(int i = 0; i < e1.data.size(); i++) {
            std::vector<size_t> multi_dim = e1.flat_index_to_mult_dim(i);
            size_t e1_index = e1.mult_dim_to_flat_index(multi_dim);
            size_t e2_index = e2.mult_dim_to_flat_index(multi_dim);
            result_data[i] = e1.data[e1_index] / e2.data[e2_index];
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

        auto start_total = std::chrono::high_resolution_clock::now();
        auto start_compact = std::chrono::high_resolution_clock::now();

        // make matrices compact to have better caching properties
        e1.compact();
        e2.compact();
        auto end_compact = std::chrono::high_resolution_clock::now();
        double compact_time = std::chrono::duration<double>(end_compact - start_compact).count();

        // (m,n) @ (n,p) => (m,p)
        size_t e1_rows = e1.shape[0];
        size_t e1_cols = e1.shape[e1_last_dim];
        size_t e2_cols = e2.shape[e2_last_dim];

        std::vector<size_t> result_shape = {e1_rows, e2_cols};
        std::vector<T> result_data(e1_rows * e2_cols, 0);

        auto start_tiling = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for collapse(2) schedule(dynamic)
        for(size_t block_x = 0; block_x < e1_rows; block_x += TILE) {
            for(size_t block_y = 0; block_y < e2.shape[e2_last_dim]; block_y += TILE) {
                tile_compute(e1.data, e2.data, result_data, block_x, block_y, e1_cols, e2_cols, e1_rows);
            }
        }

        auto end_tiling = std::chrono::high_resolution_clock::now();
        double tiling_time = std::chrono::duration<double>(end_tiling - start_tiling).count();

        auto end_total = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(end_total - start_total).count();

        std::cout << "Compact Time: " << compact_time << " seconds\n";
        std::cout << "Tiling Time: " << tiling_time << " seconds\n";
        std::cout << "Total Time: " << total_time << " seconds\n";

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

    /**
     * description: result = tensor / scalar
     * input: tensor: Tensor, scalar: T
     * output: result: Tensor
    **/
    Tensor<T> scalar_exp(Tensor<T>& tensor, T scalar) {
        std::vector<T> result_data(tensor.data.size());
        for(int i = 0; i < tensor.data.size(); i++) {
            result_data[i] = pow(tensor.data[i], scalar);
        }
        return Tensor<T>::initialize(result_data, tensor.shape);
    }

private:
    void tile_compute(std::vector<T>& e1, std::vector<T>& e2, std::vector<T>& res,
                      size_t block_x, size_t block_y, size_t e1_cols, size_t e2_cols, size_t e1_rows) {

        // adjust for when SIZE % TILE != 0
        size_t tile_height = std::min(TILE, e1_rows - block_x);
        size_t tile_width = std::min(TILE, e2_cols - block_y);

        for (size_t j = 0; j < tile_width; j++) {
            for (size_t i = 0; i < tile_height; i++) {
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

/*template <typename T>
void bind_tensor(pybind11::module& m, const std::string& class_name) {
    pybind11::class_<Tensor<T>>(m, class_name.c_str())
        .def(pybind11::init<>())
        .def_readwrite("data", &Tensor<T>::data)
        .def_readwrite("shape", &Tensor<T>::shape)
        .def_readwrite("stride", &Tensor<T>::stride)
        .def_readwrite("offset", &Tensor<T>::offset)
        .def_static("initialize", &Tensor<T>::initialize, "Initialize a Tensor",
                    pybind11::arg("data"), pybind11::arg("shape"))
        .def("compact", &Tensor<T>::compact, "Compact a Tensor")
        .def("mult_dim_to_flat_index", &Tensor<T>::mult_dim_to_flat_index);
}*/

template <typename T>
void bind_operations(pybind11::module& m, const std::string& class_name) {
    py::class_<CPUBackend<T>>(m, class_name.c_str())
        .def(py::init<>())
        .def("ewise_add", &CPUBackend<T>::ewise_add)
        .def("ewise_sub", &CPUBackend<T>::ewise_sub)
        .def("ewise_exp", &CPUBackend<T>::ewise_exp)
        .def("ewise_div", &CPUBackend<T>::ewise_div)
        .def("ewise_mul", &CPUBackend<T>::ewise_mul)
        .def("mat_mul", &CPUBackend<T>::tiled_mat_mul)
        .def("scalar_add", &CPUBackend<T>::scalar_add)
        .def("scalar_sub", &CPUBackend<T>::scalar_sub)
        .def("scalar_mul", &CPUBackend<T>::scalar_mul)
        .def("scalar_div", &CPUBackend<T>::scalar_div)
        .def("scalar_exp", &CPUBackend<T>::scalar_exp);
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
