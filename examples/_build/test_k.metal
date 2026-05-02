#include <metal_stdlib>
using namespace metal;

kernel void test_k(
    device float* v0 [[buffer(0)]],
    device float* v1 [[buffer(1)]],
    uint _sigt [[simdgroup_index_in_threadgroup]],
    uint _sgptg [[simdgroups_per_threadgroup]],
    uint _tisg [[thread_index_in_simdgroup]]
) {
    uint v2 = _tisg;
    uint v3 = _sigt;
    uint v4 = _sgptg;
    float v5 = -1.000000e+30;
    float v6 = static_cast<float>(v5);
    float v7 = 0.000000e+00;
    float v8 = static_cast<float>(v7);
    float v9 = 0.000000e+00;
    float v10 = static_cast<float>(v9);
    int v11 = 0;
    int v12 = 4;
    int v13 = 1;
    uint v14 = (uint)v11;
    uint v15 = (uint)v12;
    uint v16 = (uint)v13;
    float v17 = v8;
    float v18 = v6;
    float v19 = v10;
    for (int v20 = v14; v20 < v15; v20 += v16) {
    int v21 = 64;
    uint v22 = (uint)v21;
    uint v23 = v20 * v22;
    int v24 = 2;
    uint v25 = (uint)v24;
    uint v26 = v2 * v25;
    uint v27 = v23 + v26;
    float v28 = v0[v27];
    float v29 = fmax(v18, v28);
    float v30 = v18 - v29;
    float v31 = exp(v30);
    float v32 = v28 - v29;
    float v33 = exp(v32);
    float v34 = fma(v17, v31, v33);
    float v35 = v19 * v31;
    float v36 = fma(v28, v33, v35);
    v17 = v34;
    v18 = v29;
    v19 = v36;
    }
    threadgroup float v37[8];
    int v38 = 0;
    bool v39 = v2 == v38;
    if (v39) {
    v37[v3] = v18;
    }
    int v40 = 4;
    uint v41 = (uint)v40;
    uint v42 = v41 + v3;
    if (v39) {
    v37[v42] = v17;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool v43 = v2 < v4;
    float v44 = -1.000000e+30;
    float v45 = v37[v2];
    float v46 = select(v44, v45, v43);
    float v47 = simd_max(v46);
    float v48 = v18 - v47;
    float v49 = exp(v48);
    float v50 = v19 * v49;
    threadgroup float v51[1];
    int v52 = 0;
    bool v53 = v3 == v52;
    if (v53) {
    int v54 = 4;
    uint v55 = (uint)v54;
    uint v56 = v55 + v2;
    bool v57 = v2 < v4;
    float v58 = 0.000000e+00;
    float v59 = v37[v56];
    float v60 = select(v58, v59, v57);
    bool v61 = v2 < v4;
    float v62 = -1.000000e+30;
    float v63 = v37[v2];
    float v64 = select(v62, v63, v61);
    float v65 = v64 - v47;
    float v66 = exp(v65);
    float v67 = v60 * v66;
    float v68 = simd_sum(v67);
    bool v69 = v2 == v52;
    if (v69) {
    uint v70 = (uint)v52;
    v51[v70] = v68;
    }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float v71[128];
    int v72 = 32;
    uint v73 = (uint)v72;
    uint v74 = v3 * v73;
    uint v75 = v2 + v74;
    v71[v75] = v50;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    int v76 = 0;
    bool v77 = v3 == v76;
    if (v77) {
    float v78 = v71[v2];
    uint v79 = (uint)v72;
    uint v80 = v2 + v79;
    float v81 = v71[v80];
    float v82 = v78 + v81;
    int v83 = 64;
    uint v84 = (uint)v83;
    uint v85 = v2 + v84;
    float v86 = v71[v85];
    float v87 = v82 + v86;
    int v88 = 96;
    uint v89 = (uint)v88;
    uint v90 = v2 + v89;
    float v91 = v71[v90];
    float v92 = v87 + v91;
    uint v93 = (uint)v76;
    float v94 = v51[v93];
    float v95 = v92 / v94;
    v1[v2] = v95;
    }
}

