#include <metal_stdlib>
using namespace metal;
//#include "tensor.h"

// half is a special data type that represents floats in 16 bits
kernel void simple_shader(device half* input [[ buffer(0) ]],
                          device half* output [[ buffer(1) ]],
                          uint id [[ thread_position_in_grid ]]) {
    output[id] = input[id] * 2.0;
}

