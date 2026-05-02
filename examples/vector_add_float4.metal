#include <metal_stdlib>
using namespace metal;

kernel void vector_add_naive(
    device const float4* A [[buffer(0)]],
    device const float4* B [[buffer(1)]],
    device float4* C [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    float4 _v0 = A[tid];
    float4 _v1 = B[tid];
    float4 _v2 = _v0 + _v1;
    C[tid] = _v2;
}
