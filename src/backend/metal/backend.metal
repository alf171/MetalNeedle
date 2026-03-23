#include <metal_stdlib>
using namespace metal;

kernel void metal_ewise_add(device const float* inA, device const float* inB, device float* result, uint index [[thread_position_in_grid]]) {
    result[index] = inA[index] + inB[index];
}

kernel void metal_ewise_sub(device const float* inA, device const float* inB, device float* result, uint index [[thread_position_in_grid]]) {
    result[index] = inA[index] - inB[index];
}

kernel void metal_ewise_mul(device const float* inA, device const float* inB, device float* result, uint index [[thread_position_in_grid]]) {
    result[index] = inA[index] * inB[index];
}

kernel void metal_scalar_mul(device const float* inA, constant float& scalar [[buffer(1)]], device float* result [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    result[index] = inA[index] * scalar;
}

kernel void metal_ewise_div(device const float* inA, device const float* inB, device float* result, uint index [[thread_position_in_grid]]) {
    result[index] = inA[index] / inB[index];
}

kernel void metal_mat_mul(device const float* inA [[buffer(0)]],
                          device const float* inB [[buffer(1)]],
                          constant uint& rows_a [[buffer(2)]],
                          constant uint& cols_b [[buffer(3)]],
                          constant uint& shared_dim [[buffer(4)]],
                          device float* result [[buffer(5)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint col = gid.x;
    uint row = gid.y;

    if (row >= rows_a || col >= cols_b) {
        return;
    }

    float acc = 0.0f;
    for (uint k = 0; k < shared_dim; ++k) {
        acc += inA[row * shared_dim + k] * inB[k * cols_b + col];
    }
    result[row * cols_b + col] = acc;
}
