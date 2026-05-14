#include <metal_stdlib>
using namespace metal;

kernel void jacobi_step(
    device float* v0 [[buffer(0)]],
    device float* v1 [[buffer(1)]],
    uint3 _tpt [[thread_position_in_threadgroup]],
    uint3 _tgpg [[threadgroup_position_in_grid]]
) {
    uint v2 = 2;
    uint v3 = 17;
    uint v4 = 307;
    uint v5 = 15;
    uint v6 = 0;
    uint v7 = 18;
    uint v8 = 512;
    uint v9 = 1;
    uint v10 = 16;
    float v11 = 2.500000e-01;
    float v12 = 0.000000e+00;
    uint v13 = _tgpg.x;
    uint v14 = _tgpg.y;
    uint v15 = _tpt.x;
    uint v16 = _tpt.y;
    uint v17 = v13 * v10;
    uint v18 = v17 + v15;
    uint v19 = v14 * v10;
    uint v20 = v19 + v16;
    uint v21 = v15 + v9;
    uint v22 = v16 + v9;
    threadgroup float v23[324];
    uint v24 = v20 * v8;
    uint v25 = v24 + v18;
    float v26 = v0[v25];
    uint v27 = v22 * v7;
    uint v28 = v27 + v21;
    v23[v28] = v26;
    bool v29 = v16 == v6;
    if (v29) {
    uint v30 = v20 - v9;
    float v31 = static_cast<float>(v12);
    uint v32 = v30 * v8;
    uint v33 = v32 + v18;
    float v34 = v0[v33];
    bool v35 = v30 >= v6;
    float v36 = select(v31, v34, v35);
    v23[v21] = v36;
    }
    bool v37 = v16 == v5;
    if (v37) {
    uint v38 = v20 + v9;
    float v39 = static_cast<float>(v12);
    uint v40 = v38 * v8;
    uint v41 = v40 + v18;
    float v42 = v0[v41];
    bool v43 = v38 < v8;
    float v44 = select(v39, v42, v43);
    uint v45 = v15 + v4;
    v23[v45] = v44;
    }
    bool v46 = v15 == v6;
    if (v46) {
    uint v47 = v18 - v9;
    float v48 = static_cast<float>(v12);
    uint v49 = v24 + v47;
    float v50 = v0[v49];
    bool v51 = v47 >= v6;
    float v52 = select(v48, v50, v51);
    v23[v27] = v52;
    }
    bool v53 = v15 == v5;
    if (v53) {
    uint v54 = v18 + v9;
    float v55 = static_cast<float>(v12);
    uint v56 = v24 + v54;
    float v57 = v0[v56];
    bool v58 = v54 < v8;
    float v59 = select(v55, v57, v58);
    uint v60 = v27 + v3;
    v23[v60] = v59;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint v61 = v16 * v7;
    uint v62 = v61 + v21;
    float v63 = v23[v62];
    uint v64 = v16 + v2;
    uint v65 = v64 * v7;
    uint v66 = v65 + v21;
    float v67 = v23[v66];
    uint v68 = v27 + v15;
    float v69 = v23[v68];
    uint v70 = v15 + v2;
    uint v71 = v27 + v70;
    float v72 = v23[v71];
    float v73 = v63 + v67;
    float v74 = v73 + v72;
    float v75 = v74 + v69;
    float v76 = v75 * v11;
    v1[v25] = v76;
}

