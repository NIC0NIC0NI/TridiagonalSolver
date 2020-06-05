#ifndef HELPERS_H_INCLUDED
#define HELPERS_H_INCLUDED 1

#include <cstddef>
#include <algorithm>
#include <mkl.h>
#include "tridiagonal.h"
#include "threading.h"

template<typename T, std::size_t N>
std::size_t arr_len(T(&)[N]) { return N; }

template<typename val_t>
void init_batch(const val_t *a, const val_t *b, const val_t *c, const val_t *d, \
                val_t *a_bat, val_t *b_bat, val_t *c_bat, val_t *d_bat, int size, int batch) {
    PARALLEL_FOR_BEGIN(i, 0, batch)
        std::copy(a, a + size, a_bat + i * size);
        std::copy(b, b + size, b_bat + i * size);
        std::copy(c, c + size, c_bat + i * size);
        std::copy(d, d + size, d_bat + i * size);
    PARALLEL_FOR_END
}

template<typename val_t>
struct solver_t  {
    void (*routine)(val_t*, val_t*, val_t*, val_t*, val_t*, int, int);
    const char *name;
    int extra_buffer_size;
};

/* A wrapper of different routines for single and double precisions */
template<typename val_t> struct selector;

template<> struct selector<float> { 
    static constexpr solver_t<float> solvers[] = {
        {MKL_single, "MKL sgtsv", 0},
        {Thomas_single, "Thomas", 0},
        {CR_single, "CR", 4},
        {PCR4_single, "PCR-pThomas", 0},
        {PCR3_single, "PCR-half-pThomas", 0},
        {WM4_single, "WM-pGE", 0},
        {WM3_single, "WM-half-pGE", 0}
    };
};

template<> struct selector<double> { 
    static constexpr solver_t<double> solvers[] = {
        {MKL_double, "MKL dgtsv", 0},
        {Thomas_double, "Thomas", 0},
        {CR_double, "CR", 4},
        {PCR3_double, "PCR-pThomas", 0},
        {PCR2_double, "PCR-half-pThomas", 0},
        {WM3_double, "WM-pGE", 0},
        {WM2_double, "WM-half-pGE", 0}
    };
};


inline int check(const float *a, const float *b, int size, float *err) { 
    return check_single(a, b, size, err);
}
inline int check(const double *a, const double *b, int size, double *err) { 
    return check_double(a, b, size, err);
}

inline void MKL_rand_uniform(VSLStreamStatePtr stream, int n, float *x, float lower, float upper) {
    vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, x, lower, upper);
}
inline void MKL_rand_uniform(VSLStreamStatePtr stream, int n, double *x, double lower, double upper) {
    vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, x, lower, upper);
}


inline void MKL_rand_normal(VSLStreamStatePtr stream, int n, float *x, float mean, float stddev) {
    vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream, n, x, mean, stddev);
}
inline void MKL_rand_normal(VSLStreamStatePtr stream, int n, double *x, double mean, double stddev) {
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, stream, n, x, mean, stddev);
}
    
inline void MKL_rcondest(float *dl, float *dd, float *du, float *du2, int *ipiv, \
            int n, float norm1, float norminf, float *rcond) {
    LAPACKE_sgttrf(n, dl, dd, du, du2, ipiv);
    LAPACKE_sgtcon('1', n, dl, dd, du, du2, ipiv, norm1, rcond);
    LAPACKE_sgtcon('I', n, dl, dd, du, du2, ipiv, norminf, rcond + 1);
}

inline void MKL_rcondest(double *dl, double *dd, double *du, double *du2, int *ipiv, \
            int n, double norm1, double norminf, double *rcond) {
    LAPACKE_dgttrf(n, dl, dd, du, du2, ipiv);
    LAPACKE_dgtcon('1', n, dl, dd, du, du2, ipiv, norm1, rcond);
    LAPACKE_dgtcon('I', n, dl, dd, du, du2, ipiv, norminf, rcond + 1);
}

#endif