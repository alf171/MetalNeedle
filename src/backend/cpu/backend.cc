#include <algorithm>
#include <arm_neon.h>
#include <chrono>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <random>
#include <vector>
#include "tensor.h"

namespace py = pybind11;
const size_t TILE = 128;

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

        size_t e1_rows = e1.shape[0];
        size_t e1_cols = e1.shape[e1_last_dim];
        size_t e2_rows = e2.shape[0];
        size_t e2_cols = e2.shape[e2_last_dim];

        // (m,n) @ (n,p) => (m,p)
        // TODO: this doesn't support dim > 2
        std::vector<size_t> result_shape = {e1_rows, e2_cols};
        std::vector<T> result_data(e1_rows * e2_cols, 0);

        std::vector<T> e2_transposed(e2.data.size());
        #pragma omp parallel for collapse(2) schedule(static)
        for (size_t i = 0; i < e2_rows; i++) {
            for (size_t j = 0; j < e2_cols; j++) {
                e2_transposed[j * e2_rows + i] = e2.data[i * e2_cols + j];
            }
        }

//        if constexpr (std::is_same<T, float32_t>::value) {
////            #pragma omp parallel for collapse(2) schedule(static)
//            for(size_t block_x = 0; block_x < e1_rows; block_x += TILE) {
//                for(size_t block_y = 0; block_y < e2_cols; block_y += TILE) {
//                    simd_tile_compute(e1.data, e2_transposed, result_data,
//                                        block_x, block_y, e1_cols, e2_rows, e1_rows);
//                }
//            }
//        } else {
    #pragma omp parallel for collapse(2) schedule(static)
    for(size_t block_x = 0; block_x < e1_rows; block_x += TILE) {
        for(size_t block_y = 0; block_y < e2.shape[e2_last_dim]; block_y += TILE) {
            tile_compute(e1.data, e2.data, result_data, block_x, block_y, e1_cols, e2_cols, e1_rows);
        }
    }
//    }

        Tensor<T> result_tensor = Tensor<T>::initialize(result_data, result_shape);
        return result_tensor;
    }

    /**
     * description: result = e1 + scalar for all e1 in tensor
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

    /**
     * description: result = log(tensor)
     * input: tensor: Tensor
     * output: result: Tensor
    **/
    Tensor<T> log(Tensor<T>& tensor) {
        std::vector<T> result_data(tensor.data.size());
        for(int i = 0; i < tensor.data.size(); i++) {
            result_data[i] = std::log(tensor.data[i]);
        }
        return Tensor<T>::initialize(result_data, tensor.shape);
    }

    /**
     * description: sum across axes into a new matrix
     * input: tensor we are operating on
     * output: result: Tensor
    **/
    Tensor<T> sum(Tensor<T>& tensor, std::vector<size_t> axes) {
        std::vector<size_t> reduced_shape;
        size_t res_size = 1;

        for (size_t idx = 0; idx < tensor.shape.size(); idx++) {
            if (std::find(axes.begin(), axes.end(), idx) == axes.end()) {
                reduced_shape.push_back(tensor.shape[idx]);
                res_size *= tensor.shape[idx];
            }
        }

        // if we sum across all dims, shape should be [1] instead of []
        bool reduced_all = (axes.size() == tensor.shape.size());
        if (reduced_all) {
            reduced_shape = {1};
        }

        std::vector<T> result_data(res_size, T(0));
        Tensor<T> result = Tensor<T>::initialize(result_data, reduced_shape);

        for(size_t i = 0; i < tensor.data.size(); i++) {

            std::vector<size_t> multi_dim = tensor.flat_index_to_mult_dim(i);

            std::vector<size_t> reduced_index;
            for(size_t j = 0; j < (size_t)multi_dim.size(); j++) {
                if(std::find(axes.begin(), axes.end(), j) == axes.end()) {
                    reduced_index.push_back(multi_dim[j]);
                }
            }
            size_t flat_index = reduced_all ? 0 : result.mult_dim_to_flat_index(reduced_index);
            result.data[flat_index] += tensor.data[i];
        }
        return result;
    }

private:
    void tile_compute(const std::vector<T>& e1, const std::vector<T>& e2, std::vector<T>& res,
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
                #pragma omp atomic
                res[index_res] += tmp_sum;
            }
        }
    }

    // works exclusively for float32
    void simd_tile_compute(const std::vector<T>& e1, const std::vector<T>& e2, std::vector<T>& res,
                      const size_t block_x, const size_t block_y, const size_t e1_cols, const size_t e1_rows,
                      const size_t e2_cols) {
        size_t tile_height = std::min(TILE, e1_rows - block_x);
        size_t tile_width = std::min(TILE, e2_cols - block_y);

        for (size_t i = 0; i < tile_height; i++) {
            for (size_t j = 0; j < tile_width; j++) {
                T tmp_sum = 0;
                float32x4_t acc1 = vdupq_n_f32(0.0);
                float32x4_t acc2 = vdupq_n_f32(0.0);

                size_t k = 0;
                for (; k < e1_cols - 7; k += 8) {
                    size_t index_e1 = (block_x + i) * e1_cols + k;
                    size_t index_e2 = k * e2_cols + (block_y + j);
                    float32x4_t e1_vec1 = vld1q_f32(&e1[index_e1]);
                    float32x4_t e2_vec1 = vld1q_f32(&e2[index_e2]);
                    float32x4_t e1_vec2 = vld1q_f32(&e1[index_e1+4]);
                    float32x4_t e2_vec2 = vld1q_f32(&e2[index_e2+4]);
                    acc1 = vmlaq_f32(acc1, e1_vec1, e2_vec1);
                    acc2 = vmlaq_f32(acc2, e1_vec2, e2_vec2);
                }
                tmp_sum = vaddvq_f32(acc1) + vaddvq_f32(acc2);

                for (; k < e1_cols; k++) {
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
        .def("scalar_exp", &CPUBackend<T>::scalar_exp)
        .def("log", &CPUBackend<T>::log)
        .def("sum", &CPUBackend<T>::sum);
}

// could consider moving this
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
