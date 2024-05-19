#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <initializer_list>

namespace py = pybind11;
template<typename T>

class CPUBackend {
public:
    CPUBackend() = default;
    
    // CPUBackend(std::initializer_list<T> data, std::initializer_list<size_t> size)

    CPUBackend(std::initializer_list<T> data, std::initializer_list<size_t> shape): data(data), shape(shape) {
        if (data.size() != calculate_size(shape)) {
            throw std::invalid_argument("Data size does not match shape dimensions.");
        }
    }

    std::vector<size_t> get_shape(){
        return shape;
    }


private:
    std::vector<T> data;
    std::vector<size_t> shape;
    // will use these two later
    std::vector<size_t> stride;
    size_t offset;

    size_t calculate_size(std::initializer_list<size_t> shape){
        size_t r_size = 1;
        for(size_t shape: shape) { 
            r_size *= shape;
        }
        return r_size;
    }


    // Make our array contiguous. Many matrix operation are implemented by manipulating
    // shape, stride, and offset. However, some operations requrie our matrix to compact..
    void compact(std::vector<int> input, std::vector<int32_t> shape, std::vector<int32_t> stride, size_t offset) {

    }
};


PYBIND11_MODULE(cpu_backend, m) {
    py::class_<CPUBackend<double>>(m, "CPUBackend")
        .def(py::init<>())
        .def(py::init<const std::vector<double>&, const std::vector<size_t>&>());
}