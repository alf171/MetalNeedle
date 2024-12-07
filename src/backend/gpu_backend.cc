// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <vector>
// #include <iostream>

// namespace py = pybind11;
// // this number might have to be tuned to match width of asm instruction
// #define TILE 8

// template<typename T>
// struct Tensor {
//     std::vector<T> data;
//     std::vector<size_t> shape;
//     std::vector<size_t> stride;
//     size_t offset;

//   static Tensor<T> initialize(const std::vector<T>& data, const std::vector<size_t>& shape){
//         if (data.size() != calculate_size(shape)) {
//             throw std::invalid_argument("Data size does not match shape dimensions.");
//         }

//         Tensor<T> tensor;
//         tensor.data = data;
//         tensor.shape = shape;
//         tensor.stride = calculate_stride(shape); 
//         tensor.offset = 0;
//         return tensor;
//     }

// private:
//     static size_t calculate_size(const std::vector<size_t>& shape) {
//         size_t r_size = 1;
//         for (size_t s : shape) { 
//             r_size *= s;
//         }
//         return r_size;
//     }

//     static std::vector<size_t> calculate_stride(const std::vector<size_t>& shape) {
//         std::vector<size_t> stride(shape.size());
//         size_t product = 1;
//         for(int i = shape.size() - 1; i >= 0; --i) {
//             stride[i] = product;
//             product *= shape[i];
//         }
//         return stride;
//     }

// public:
//     size_t mult_dim_to_flat_index(const std::vector<size_t>& dimension) const {
//         size_t result = offset;
//         for (int i = 0; i < shape.size(); i++) {
//             result += dimension[i] * stride[i];
//         }
//         return result;
//     }


//     std::vector<size_t> flat_index_to_mult_dim(const size_t index) const {
//         std::vector<size_t> result(shape.size());
//         size_t current = index;
//         for(int i = shape.size() - 1; i >= 0; --i) {
//             result[i] = current % shape[i];
//             current /= shape[i];
//         }
//         return result;
//     }
// };


// template<typename T>
// class GPUBackend {
// public:
//     GPUBackend() = default;

//     Tensor<T> ewise_add(Tensor<T>& e1, Tensor<T>& e2) {

//     }

//     Tensor<T> ewise_mul(Tensor<T>& e1, Tensor<T>& e2) {

//     }

//     Tensor<T> tiled_mat_mul(Tensor<T>& e1, Tensor<T>& e2) {

//     }

//     Tensor<T> naive_mat_mult(Tensor<T> e1, Tensor<T> e2) {

//     }

// private:
//     // can only be applied to matricies that are size (TILE, TILE)
//     void tile_compute(std::vector<T>& e1, std::vector<T>& e2, std::vector<T>& res, size_t block_x, size_t block_y, size_t e1_col, size_t e2_col) {

//     }    



//     // // Make our array contiguous. Many matrix operation are implemented by manipulating
//     // // shape, stride, and offset. However, some operations requrie our matrix to be compact..
//     // void compact(std::vector<int> input, std::vector<int32_t> shape, std::vector<int32_t> stride, size_t offset) {}
// };

// // template <typename T>
// // void bind_tensor(pybind11::module& m, const std::string& class_name) {
// //     pybind11::class_<Tensor<T>>(m, class_name.c_str())
// //         .def(pybind11::init<>())
// //         .def_readwrite("data", &Tensor<T>::data)
// //         .def_readwrite("shape", &Tensor<T>::shape)
// //         .def_readwrite("stride", &Tensor<T>::stride)
// //         .def_readwrite("offset", &Tensor<T>::offset)
// //         .def_static("initialize", &Tensor<T>::initialize, "Initialize a Tensor",
// //                     pybind11::arg("data"), pybind11::arg("shape"))
// //         .def("mult_dim_to_flat_index", &Tensor<T>::mult_dim_to_flat_index);
// // }

// template <typename T>
// void bind_operations(pybind11::module& m, const std::string& class_name) {
//     py::class_<GPUBackend<T>>(m, class_name.c_str())
//         .def(py::init<>())
//         .def("ewise_add", &GPUBackend<T>::ewise_add)
//         .def("ewise_mul", &GPUBackend<T>::ewise_mul)
//         .def("mat_mul", &GPUBackend<T>::tiled_mat_mul);
// }


// void bind_gpu(py::module &m) {
//     auto gpu = m.def_submodule("gpu");
//     // operations
//     bind_operations<int32_t>(gpu, "IntOperation");
//     bind_operations<int64_t>(gpu, "LongOperation");
//     bind_operations<float>(gpu, "FloatOperation");
//     bind_operations<double>(gpu, "DoubleOperation");
//     // data
//     // bind_tensor<int32_t>(gpu, "IntTensor");
//     // bind_tensor<int64_t>(gpu, "LongTensor");
//     // bind_tensor<float>(gpu, "FloatTensor");
//     // bind_tensor<double>(gpu, "DoubleTensor");
// }
