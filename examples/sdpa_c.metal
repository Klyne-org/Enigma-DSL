#include <metal_stdlib>
using namespace metal;

kernel void sdpa_kernel(
    device float* v0 [[buffer(0)]],
    device float* v1 [[buffer(1)]],
    device float* v2 [[buffer(2)]],
    device float* v3 [[buffer(3)]],
    uint3 _tpt [[thread_position_in_threadgroup]],
    uint3 _tgpg [[threadgroup_position_in_grid]],
    uint _sigt [[simdgroup_index_in_threadgroup]],
    uint _sgptg [[simdgroups_per_threadgroup]],
    uint _tisg [[thread_index_in_simdgroup]]
) {
    uint v4 = 1;
    uint v5 = 33;
    uint v6 = 0;
    float v7 = 1.250000e-01;
    float v8 = 0.000000e+00;
    int v9 = 1;
    int v10 = 0;
    float v11 = -1.000000e+30;
    int v12 = 2;
    int v13 = 2112;
    int v14 = 64;
    uint v15 = _tgpg.x;
    uint v16 = _tpt.x;
    uint v17 = _tisg;
    uint v18 = _sigt;
    uint v19 = _sgptg;
    int v20 = static_cast<int>(v14);
    int v21 = static_cast<int>(v13);
    int v22 = static_cast<int>(v12);
    uint v23 = (uint)v22;
    uint v24 = v15 / v23;
    uint v25 = (uint)v20;
    uint v26 = v15 * v25;
    uint v27 = v26 + v16;
    float v28 = v0[v27];
    float v29 = static_cast<float>(v11);
    float v30 = static_cast<float>(v10);
    threadgroup float v31[32];
    threadgroup float v32[1];
    float v33 = v29;
    float v34 = v30;
    float v35 = v30;
    for (int v36 = v6; v36 < v5; v36 += v4) {
    int v37 = static_cast<int>(v36);
    uint v38 = (uint)v21;
    uint v39 = v24 * v38;
    int v40 = v37 * v20;
    uint v41 = (uint)v40;
    uint v42 = v39 + v41;
    uint v43 = v42 + v16;
    float v44 = v1[v43];
    float v45 = v2[v43];
    float v46 = v28 * v44;
    float v47 = simd_sum(v46);
    bool v48 = v17 == v6;
    if (v48) {
    v31[v18] = v47;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool v49 = v17 < v19;
    float v50 = v31[v17];
    float v51 = select(v8, v50, v49);
    float v52 = simd_sum(v51);
    float v53 = v52 * v7;
    bool v54 = v16 == v6;
    if (v54) {
    v32[v6] = v53;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float v55 = v32[v6];
    float v56 = fmax(v33, v55);
    float v57 = v33 - v56;
    float v58 = exp(v57);
    float v59 = v55 - v56;
    float v60 = exp(v59);
    float v61 = v34 * v58;
    float v62 = v61 + v60;
    float v63 = v35 * v58;
    float v64 = v60 * v45;
    float v65 = v63 + v64;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v33 = v56;
    v34 = v62;
    v35 = v65;
    }
    float v66 = static_cast<float>(v9);
    float v67 = v66 / v34;
    float v68 = v35 * v67;
    v3[v27] = v68;
}

