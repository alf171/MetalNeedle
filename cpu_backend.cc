#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <initializer_list>

namespace py = pybind11;
template<typename T>

class CPUBackend {
public:
    // possible lazy init
    CPUBackend() = default;

    // used for lazy init
    void initialize(const std::vector<T>& data, const std::vector<size_t>& shape){
        if (data.size() != calculate_size(shape)) {
            throw std::invalid_argument("Data size does not match shape dimensions.");
        }
        this->data = data;
        this->shape = shape;
    }

    // default initializer 
    CPUBackend(const std::vector<T>& data, const std::vector<size_t>& shape)
        : data(data), shape(shape) {
        if (data.size() != calculate_size(shape)) {
            throw std::invalid_argument("Data size does not match shape dimensions.");
        }
    }

    void reshape(const std::vector<size_t>& new_shape){
        assert(data != NULL);
        assert(shape != NULL);

    }

    void transpose(const size_t dim1, const size_t dim2) {
        assert(data != NULL);
        assert(shape != NULL);
        shape = std::swap(shape[dim1], shape[dim2]);
    }

private:
    std::vector<T> data;
    std::vector<size_t> shape;
    // will use these two later
    std::vector<size_t> stride;
    size_t offset;

    size_t calculate_size(const std::vector<size_t>& shape) const {
        size_t r_size = 1;
        for (size_t s : shape) { 
            r_size *= s;
        }
        return r_size;
    }

    // Make our array contiguous. Many matrix operation are implemented by manipulating
    // shape, stride, and offset. However, some operations requrie our matrix to be compact..
    void compact(std::vector<int> input, std::vector<int32_t> shape, std::vector<int32_t> stride, size_t offset) {}
};


PYBIND11_MODULE(cpu_backend, m) {
    py::class_<CPUBackend<double>>(m, "CPUBackend")
        .def(py::init<>())
        .def(py::init<const std::vector<double>&, const std::vector<size_t>&>())
        .def("initialize", &CPUBackend<double>::initialize);
}