#ifndef INTRIN_H_INCLUDED
#define INTRIN_H_INCLUDED 1

#include <cuda.h>

#if defined(RCP)
#define DRCP
#define FRCP
#elif defined(PCR_APPROX)
#define DRCP_APPROX
#define FRCP_APPROX
#elif defined(DIV)
#define DDIV
#define FDIV
#elif defined(DIV_APPROX)
#define DDIV
#define FDIV_APPROX
#else  // best
#define DRCP
#define FDIV_APPROX
#endif

#if defined(DRCP)
__device__ __forceinline__ double div(double a, double b) {
    return __dmul_rn(a, __drcp_rn(b));
}
#elif defined(DRCP_APPROX)
__device__ __forceinline__ double rcp_approx(double x) {
    double y;
    asm ("rcp.approx.ftz.f64 %0, %1;" : "=d"(y) : "d"(x));
    return y;
}
__device__ __forceinline__ double div(double a, double b) {
    return __dmul_rn(a, rcp_approx(b));
}
#elif defined(DDIV)
__device__ __forceinline__ double div(double a, double b) {
    return __ddiv_rn(a, b);
}
#endif


#if defined(FRCP)
__device__ __forceinline__ float div(float a, float b) {
    return __fmul_rn(a, __frcp_rn(b));
}
#elif defined(FRCP_APPROX)
__device__ __forceinline__ float rcp_approx(float x) {
    float y;
    asm ("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
__device__ __forceinline__ float div(float a, float b) {
    return __fmul_rn(a, rcp_approx(b));
}
#elif defined(FDIV_APPROX)
__device__ __forceinline__ float div(float a, float b) {
    float c;
    asm ("div.approx.ftz.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}
#elif defined(FDIV)
__device__ __forceinline__ float div(float a, float b) {
    return __fdiv_rn(a, b);
}
#endif

__device__ __forceinline__ double maskz_div(bool is_zero, double a, double b) {
    return is_zero ? 0.0 : div(a, b);
}
__device__ __forceinline__ float maskz_div(bool is_zero, float a, float b) {
    return is_zero ? 0.0f : div(a, b);
}

__device__ __forceinline__ double2 load_2(const double *src) {
    return *reinterpret_cast<const double2*>(src);
}
__device__ __forceinline__ float2 load_2(const float *src) {
    return *reinterpret_cast<const float2*>(src);
}

__device__ __forceinline__ double4 load_4(const double *src) {
    return *reinterpret_cast<const double4*>(src);
}
__device__ __forceinline__ float4 load_4(const float *src) {
    return *reinterpret_cast<const float4*>(src);
}

#endif