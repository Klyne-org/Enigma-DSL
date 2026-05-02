#include <metal_stdlib>
using namespace metal;

kernel void add_kernel_tv(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    uint tidx [[thread_position_in_threadgroup]],
    uint bidx [[threadgroup_position_in_grid]]
) {
    uint _c64 = 64;
    uint _v0 = bidx % _c64;
    uint _v1 = bidx / _c64;
    uint _c16384 = 16384;
    uint _v2 = _v0 * _c16384;
    uint _c4 = 4;
    uint _v3 = _v1 % _c4;
    uint _v4 = _v1 / _c4;
    uint _c256 = 256;
    uint _v5 = _v3 * _c256;
    uint _v6 = _v2 + _v5;
    uint _v7 = bidx % _c64;
    uint _v8 = bidx / _c64;
    uint _v9 = _v7 * _c16384;
    uint _v10 = _v8 % _c4;
    uint _v11 = _v8 / _c4;
    uint _v12 = _v10 * _c256;
    uint _v13 = _v9 + _v12;
    uint _v14 = bidx % _c64;
    uint _v15 = bidx / _c64;
    uint _v16 = _v14 * _c16384;
    uint _v17 = _v15 % _c4;
    uint _v18 = _v15 / _c4;
    uint _v19 = _v17 * _c256;
    uint _v20 = _v16 + _v19;
    uint _v21 = tidx % _c64;
    uint _v22 = tidx / _c64;
    uint _v23 = _v21 * _c4;
    uint _v24 = _v22 % _c4;
    uint _v25 = _v22 / _c4;
    uint _c4096 = 4096;
    uint _v26 = _v24 * _c4096;
    uint _v27 = _v23 + _v26;
    uint _v28 = _v6 + _v27;
    uint _v29 = tidx % _c64;
    uint _v30 = tidx / _c64;
    uint _v31 = _v29 * _c4;
    uint _v32 = _v30 % _c4;
    uint _v33 = _v30 / _c4;
    uint _v34 = _v32 * _c4096;
    uint _v35 = _v31 + _v34;
    uint _v36 = _v13 + _v35;
    uint _v37 = tidx % _c64;
    uint _v38 = tidx / _c64;
    uint _v39 = _v37 * _c4;
    uint _v40 = _v38 % _c4;
    uint _v41 = _v38 / _c4;
    uint _v42 = _v40 * _c4096;
    uint _v43 = _v39 + _v42;
    uint _v44 = _v20 + _v43;
    float4 _v45_g0_v0 = *reinterpret_cast<device const float4*>(&A[_v28]);
    float4 _v45_g1_v0 = *reinterpret_cast<device const float4*>(&A[_v28 + 1024]);
    float4 _v45_g2_v0 = *reinterpret_cast<device const float4*>(&A[_v28 + 2048]);
    float4 _v45_g3_v0 = *reinterpret_cast<device const float4*>(&A[_v28 + 3072]);
    float4 _v46_g0_v0 = *reinterpret_cast<device const float4*>(&B[_v36]);
    float4 _v46_g1_v0 = *reinterpret_cast<device const float4*>(&B[_v36 + 1024]);
    float4 _v46_g2_v0 = *reinterpret_cast<device const float4*>(&B[_v36 + 2048]);
    float4 _v46_g3_v0 = *reinterpret_cast<device const float4*>(&B[_v36 + 3072]);
    float4 _v47_g0_v0 = _v45_g0_v0 + _v46_g0_v0;
    float4 _v47_g1_v0 = _v45_g1_v0 + _v46_g1_v0;
    float4 _v47_g2_v0 = _v45_g2_v0 + _v46_g2_v0;
    float4 _v47_g3_v0 = _v45_g3_v0 + _v46_g3_v0;
    *reinterpret_cast<device float4*>(&C[_v44]) = _v47_g0_v0;
    *reinterpret_cast<device float4*>(&C[_v44 + 1024]) = _v47_g1_v0;
    *reinterpret_cast<device float4*>(&C[_v44 + 2048]) = _v47_g2_v0;
    *reinterpret_cast<device float4*>(&C[_v44 + 3072]) = _v47_g3_v0;
}
