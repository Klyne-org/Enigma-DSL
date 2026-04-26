#include <metal_stdlib>
using namespace metal;

kernel void vector_add_naive(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    float _v0 = A[tid];
    float _v1 = B[tid];
    float _v2 = _v0 + _v1;
    C[tid] = _v2;
}
