#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm(
    device const float4* input   [[buffer(0)]],
    device const float4* weight  [[buffer(1)]],
    device float4*       output  [[buffer(2)]],
    uint gid   [[threadgroup_position_in_grid]],
    uint tid   [[thread_position_in_threadgroup]],
    uint tsize [[threads_per_threadgroup]]
) {
    const uint N = 4096;
    const uint N4 = N / 4;
    const float eps = 1e-5f;
    threadgroup float shared[32];

    device const float4* row_in  = input  + gid * N4;
    device float4*       row_out = output + gid * N4;

    float4 sumsq4 = 0.0f;
    for (uint i = tid; i < N4; i += tsize) {
        float4 v = row_in[i];
        sumsq4 = fma(v, v, sumsq4);
    }
    float sumsq = sumsq4.x + sumsq4.y + sumsq4.z + sumsq4.w;

    sumsq = simd_sum(sumsq);
    uint simd_lane = tid % 32;
    uint simd_idx  = tid / 32;
    if (simd_lane == 0) shared[simd_idx] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float total = 0.0f;
    if (simd_idx == 0) {
        total = (simd_lane < (tsize / 32)) ? shared[simd_lane] : 0.0f;
        total = simd_sum(total);
    }
    if (tid == 0) shared[0] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = shared[0];

    float scale = rsqrt(total / float(N) + eps);
    for (uint i = tid; i < N4; i += tsize) {
        row_out[i] = row_in[i] * scale * weight[i];
    }
}
