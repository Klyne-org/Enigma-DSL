#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

struct sdpa_args {
    uint32_t qkv_dim;
    uint32_t num_kv_tokens;
    uint32_t kv_stride;
    uint32_t window;
};

struct control_t {
    uint32_t abort;
};

kernel void sdpa_gptoss(
    device const sdpa_args& args [[ buffer(0) ]],
    const device float* q [[ buffer(1) ]],
    const device float* kv [[ buffer(2) ]],
    const device float* s [[ buffer(3) ]],
    device float* output [[ buffer(4) ]],
    uint2 gid [[threadgroup_position_in_grid]],
    uint simdgroup_tid [[thread_index_in_simdgroup]],
    uint simdgroup_idx [[simdgroup_index_in_threadgroup]],
    uint num_simdgroups [[simdgroups_per_threadgroup]])
{
    const uint num_q_heads = 64;
    const uint head_dim = 64;
    const uint qmul = 8;
    const uint token_stride = 2 * head_dim;

    const uint qt = gid.x;
    const uint h = gid.y;

    q += qt * args.qkv_dim + h * (qmul * head_dim);
    kv += h * args.kv_stride;
    output += qt * (num_q_heads * head_dim) + h * (qmul * head_dim);

    float m0=s[h*qmul+0],m1=s[h*qmul+1],m2=s[h*qmul+2],m3=s[h*qmul+3];
    float m4=s[h*qmul+4],m5=s[h*qmul+5],m6=s[h*qmul+6],m7=s[h*qmul+7];

    float l0=simdgroup_idx==0?1.0f:0.0f,l1=l0,l2=l0,l3=l0,l4=l0,l5=l0,l6=l0,l7=l0;
    float2 o0=0,o1=0,o2=0,o3=0,o4=0,o5=0,o6=0,o7=0;

    float2 q0=reinterpret_cast<const device float2*>(q+0*head_dim)[simdgroup_tid];
    float2 q1=reinterpret_cast<const device float2*>(q+1*head_dim)[simdgroup_tid];
    float2 q2=reinterpret_cast<const device float2*>(q+2*head_dim)[simdgroup_tid];
    float2 q3=reinterpret_cast<const device float2*>(q+3*head_dim)[simdgroup_tid];
    float2 q4=reinterpret_cast<const device float2*>(q+4*head_dim)[simdgroup_tid];
    float2 q5=reinterpret_cast<const device float2*>(q+5*head_dim)[simdgroup_tid];
    float2 q6=reinterpret_cast<const device float2*>(q+6*head_dim)[simdgroup_tid];
    float2 q7=reinterpret_cast<const device float2*>(q+7*head_dim)[simdgroup_tid];

    const uint kt_end = args.num_kv_tokens;
    const uint kt_start = simdgroup_idx;
    const device float* kv_ptr = kv + token_stride * kt_start;

    for (uint kt = kt_start; kt < kt_end; kt += num_simdgroups) {
        const float2 kval = reinterpret_cast<const device float2*>(kv_ptr)[simdgroup_tid];

        float qk0=simd_sum(dot(q0,kval)),qk1=simd_sum(dot(q1,kval));
        float qk2=simd_sum(dot(q2,kval)),qk3=simd_sum(dot(q3,kval));
        float qk4=simd_sum(dot(q4,kval)),qk5=simd_sum(dot(q5,kval));
        float qk6=simd_sum(dot(q6,kval)),qk7=simd_sum(dot(q7,kval));

        float nm0=max(m0,qk0),nm1=max(m1,qk1),nm2=max(m2,qk2),nm3=max(m3,qk3);
        float nm4=max(m4,qk4),nm5=max(m5,qk5),nm6=max(m6,qk6),nm7=max(m7,qk7);

        float a0=exp(m0-nm0),a1=exp(m1-nm1),a2=exp(m2-nm2),a3=exp(m3-nm3);
        float a4=exp(m4-nm4),a5=exp(m5-nm5),a6=exp(m6-nm6),a7=exp(m7-nm7);

        float p0=exp(qk0-nm0),p1=exp(qk1-nm1),p2=exp(qk2-nm2),p3=exp(qk3-nm3);
        float p4=exp(qk4-nm4),p5=exp(qk5-nm5),p6=exp(qk6-nm6),p7=exp(qk7-nm7);

        l0=fma(l0,a0,p0);l1=fma(l1,a1,p1);l2=fma(l2,a2,p2);l3=fma(l3,a3,p3);
        l4=fma(l4,a4,p4);l5=fma(l5,a5,p5);l6=fma(l6,a6,p6);l7=fma(l7,a7,p7);
        m0=nm0;m1=nm1;m2=nm2;m3=nm3;m4=nm4;m5=nm5;m6=nm6;m7=nm7;

        const float2 vval = reinterpret_cast<const device float2*>(kv_ptr + head_dim)[simdgroup_tid];
        kv_ptr += token_stride * num_simdgroups;
        o0=fma(vval,p0,o0*a0);o1=fma(vval,p1,o1*a1);o2=fma(vval,p2,o2*a2);o3=fma(vval,p3,o3*a3);
        o4=fma(vval,p4,o4*a4);o5=fma(vval,p5,o5*a5);o6=fma(vval,p6,o6*a6);o7=fma(vval,p7,o7*a7);
    }

    // Cross-simdgroup reduction
    threadgroup float shared_m[8 * 4];
    threadgroup float shared_l[8 * 4];

    if (simd_is_first()) {
        shared_m[0*num_simdgroups+simdgroup_idx]=m0;shared_m[1*num_simdgroups+simdgroup_idx]=m1;
        shared_m[2*num_simdgroups+simdgroup_idx]=m2;shared_m[3*num_simdgroups+simdgroup_idx]=m3;
        shared_m[4*num_simdgroups+simdgroup_idx]=m4;shared_m[5*num_simdgroups+simdgroup_idx]=m5;
        shared_m[6*num_simdgroups+simdgroup_idx]=m6;shared_m[7*num_simdgroups+simdgroup_idx]=m7;
        shared_l[0*num_simdgroups+simdgroup_idx]=l0;shared_l[1*num_simdgroups+simdgroup_idx]=l1;
        shared_l[2*num_simdgroups+simdgroup_idx]=l2;shared_l[3*num_simdgroups+simdgroup_idx]=l3;
        shared_l[4*num_simdgroups+simdgroup_idx]=l4;shared_l[5*num_simdgroups+simdgroup_idx]=l5;
        shared_l[6*num_simdgroups+simdgroup_idx]=l6;shared_l[7*num_simdgroups+simdgroup_idx]=l7;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sm0=m0,sm1=m1,sm2=m2,sm3=m3,sm4=m4,sm5=m5,sm6=m6,sm7=m7;
    if (simdgroup_tid < num_simdgroups) {
        sm0=shared_m[0*num_simdgroups+simdgroup_tid];sm1=shared_m[1*num_simdgroups+simdgroup_tid];
        sm2=shared_m[2*num_simdgroups+simdgroup_tid];sm3=shared_m[3*num_simdgroups+simdgroup_tid];
        sm4=shared_m[4*num_simdgroups+simdgroup_tid];sm5=shared_m[5*num_simdgroups+simdgroup_tid];
        sm6=shared_m[6*num_simdgroups+simdgroup_tid];sm7=shared_m[7*num_simdgroups+simdgroup_tid];
    }
    float gm0=simd_max(sm0),gm1=simd_max(sm1),gm2=simd_max(sm2),gm3=simd_max(sm3);
    float gm4=simd_max(sm4),gm5=simd_max(sm5),gm6=simd_max(sm6),gm7=simd_max(sm7);

    o0*=exp(m0-gm0);o1*=exp(m1-gm1);o2*=exp(m2-gm2);o3*=exp(m3-gm3);
    o4*=exp(m4-gm4);o5*=exp(m5-gm5);o6*=exp(m6-gm6);o7*=exp(m7-gm7);

    if (simdgroup_idx == 0) {
        l0=0;l1=0;l2=0;l3=0;l4=0;l5=0;l6=0;l7=0;
        if (simdgroup_tid < num_simdgroups) {
            l0=shared_l[0*num_simdgroups+simdgroup_tid];l1=shared_l[1*num_simdgroups+simdgroup_tid];
            l2=shared_l[2*num_simdgroups+simdgroup_tid];l3=shared_l[3*num_simdgroups+simdgroup_tid];
            l4=shared_l[4*num_simdgroups+simdgroup_tid];l5=shared_l[5*num_simdgroups+simdgroup_tid];
            l6=shared_l[6*num_simdgroups+simdgroup_tid];l7=shared_l[7*num_simdgroups+simdgroup_tid];
        }
        l0=simd_sum(l0*exp(sm0-gm0));l1=simd_sum(l1*exp(sm1-gm1));
        l2=simd_sum(l2*exp(sm2-gm2));l3=simd_sum(l3*exp(sm3-gm3));
        l4=simd_sum(l4*exp(sm4-gm4));l5=simd_sum(l5*exp(sm5-gm5));
        l6=simd_sum(l6*exp(sm6-gm6));l7=simd_sum(l7*exp(sm7-gm7));
    }

    // Output tree reduction via shared memory
    threadgroup float2 shared_o[8 * 128];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_idx > 0) {
        shared_o[0*32+simdgroup_tid+(simdgroup_idx-1)*32]=o0;
        shared_o[1*32+simdgroup_tid+(simdgroup_idx-1)*32]=o1;
        shared_o[2*32+simdgroup_tid+(simdgroup_idx-1)*32]=o2;
        shared_o[3*32+simdgroup_tid+(simdgroup_idx-1)*32]=o3;
        shared_o[4*32+simdgroup_tid+(simdgroup_idx-1)*32]=o4;
        shared_o[5*32+simdgroup_tid+(simdgroup_idx-1)*32]=o5;
        shared_o[6*32+simdgroup_tid+(simdgroup_idx-1)*32]=o6;
        shared_o[7*32+simdgroup_tid+(simdgroup_idx-1)*32]=o7;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup_idx == 0) {
        for (uint s = 0; s < num_simdgroups - 1; s++) {
            o0+=shared_o[0*32+simdgroup_tid+s*32];o1+=shared_o[1*32+simdgroup_tid+s*32];
            o2+=shared_o[2*32+simdgroup_tid+s*32];o3+=shared_o[3*32+simdgroup_tid+s*32];
            o4+=shared_o[4*32+simdgroup_tid+s*32];o5+=shared_o[5*32+simdgroup_tid+s*32];
            o6+=shared_o[6*32+simdgroup_tid+s*32];o7+=shared_o[7*32+simdgroup_tid+s*32];
        }
        reinterpret_cast<device float2*>(output+0*head_dim)[simdgroup_tid]=o0/l0;
        reinterpret_cast<device float2*>(output+1*head_dim)[simdgroup_tid]=o1/l1;
        reinterpret_cast<device float2*>(output+2*head_dim)[simdgroup_tid]=o2/l2;
        reinterpret_cast<device float2*>(output+3*head_dim)[simdgroup_tid]=o3/l3;
        reinterpret_cast<device float2*>(output+4*head_dim)[simdgroup_tid]=o4/l4;
        reinterpret_cast<device float2*>(output+5*head_dim)[simdgroup_tid]=o5/l5;
        reinterpret_cast<device float2*>(output+6*head_dim)[simdgroup_tid]=o6/l6;
        reinterpret_cast<device float2*>(output+7*head_dim)[simdgroup_tid]=o7/l7;
    }
}
