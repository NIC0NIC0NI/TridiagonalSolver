#ifndef HELPERS_H_INCLUDED
#define HELPERS_H_INCLUDED 1

#include <cstddef>
#include <algorithm>
#include <mkl.h>
#include "tridiagonal.h"

template<typename T, std::size_t N> std::size_t arr_len(T(&)[N]) { return N; }

template<typename val_t>
struct solver_t  {
    void (*routine)(val_t*, val_t*, val_t*, val_t*, int, int);
    const char *name;
    int (*init)(val_t*, val_t*, val_t*, val_t*, int, int);
    void (*final)();
    int layout;
};

/* A wrapper of different routines for single and double precisions */
template<typename val_t> struct selector;

template<> struct selector<float> { 
    static constexpr solver_t<float> solvers[] = {
        {cuSparse_single, "cuSparse sgtsv", cuSparse_init_single, cuSparse_final, 0},
        {Thomas_single, "Thomas", nullptr, nullptr, 0},
        {CR_single, "CR", CR_init_single, CR_final, 0},
        {PCR5_shmem_single, "shmem PCR-pThomas", nullptr, nullptr, 0},
        {PCR4_shmem_single, "shmem PCR-half-pThomas", nullptr, nullptr, 0},
        {PCR5_reg_single, "reg PCR-pThomas", nullptr, nullptr, 0},
        {PCR4_reg_single, "reg PCR-half-pThomas", nullptr, nullptr, 0},
        {WM5_shmem_single, "shmem WM-pGE", nullptr, nullptr, 0},
        {WM4_shmem_single, "shmem WM-half-pGE", nullptr, nullptr, 0},
        {WM5_reg_single, "reg WM-pGE", nullptr, nullptr, 0},
        {WM4_reg_single, "reg WM-half-pGE", nullptr, nullptr, 0},
        {CR_reg4_single, "reg-4 CR", nullptr, nullptr, 0},
        {CR_reg8_single, "reg-8 CR", nullptr, nullptr, 0},
        {CR_reg16_single, "reg-16 CR", nullptr, nullptr, 0}
#ifdef BPLG
        ,
        {BPLG_WM, "BPLG WM", nullptr, nullptr, 1},
        {BPLG_LF, "BPLG LF", nullptr, nullptr, 1},
        {BPLG_CR, "BPLG CR", nullptr, nullptr, 1},
        {BPLG_PCR, "BPLG PCR", nullptr, nullptr, 1}
#endif
    };
};

template<> struct selector<double> { 
    static constexpr solver_t<double> solvers[] = {
        {cuSparse_double, "cuSparse dgtsv", cuSparse_init_double, cuSparse_final, 0},
        {Thomas_double, "Thomas", nullptr, nullptr, 0},
        {CR_double, "CR", CR_init_double, CR_final, 0},
        {PCR5_shmem_double, "shmem PCR-pThomas", nullptr, nullptr, 0},
        {PCR4_shmem_double, "shmem PCR-half-pThomas", nullptr, nullptr, 0},
        {PCR5_reg_double, "reg PCR-pThomas", nullptr, nullptr, 0},
        {PCR4_reg_double, "reg PCR-half-pThomas", nullptr, nullptr, 0},
        {WM5_shmem_double, "shmem WM-pGE", nullptr, nullptr, 0},
        {WM4_shmem_double, "shmem WM-half-pGE", nullptr, nullptr, 0},
        {WM5_reg_double, "reg WM-pGE", nullptr, nullptr, 0},
        {WM4_reg_double, "reg WM-half-pGE", nullptr, nullptr, 0},
        {CR_reg4_double, "reg-4 CR", nullptr, nullptr, 0},
        {CR_reg8_double, "reg-8 CR", nullptr, nullptr, 0},
        {CR_reg16_double, "reg-16 CR", nullptr, nullptr, 0}
    };
};

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