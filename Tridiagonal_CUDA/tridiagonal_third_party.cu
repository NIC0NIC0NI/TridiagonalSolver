#include <iostream>
#include "alg/lib-BPLG/KTridiagCR.hxx"
#include "alg/lib-BPLG/KTridiagPCR.hxx"
#include "alg/lib-BPLG/KTridiagWM.hxx"
#include "alg/lib-BPLG/KTridiagLF.hxx"
#include "tridiagonal.h"

#define BPLG_CHECK_ERROR(err, errorMessage) {                         \
    if( 0 != err) {                                                   \
        std::cerr << "Cuda error: " << errorMessage << " in file '"   \
            << __FILE__ << "' in line " << __LINE__ << " : "          \
            << err << "." << std::endl;                               \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}


void BPLG_WM(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    int ret = KTridiagWM(a_gbl, 0, n_eqt, 4, n_batch);
    cudaDeviceSynchronize();
    BPLG_CHECK_ERROR(ret, "BPLG_WM")
}

void BPLG_LF(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    int ret = KTridiagLF(a_gbl, 0, n_eqt, 4, n_batch);
    cudaDeviceSynchronize();
    BPLG_CHECK_ERROR(ret, "BPLG_LF")
}

void BPLG_CR(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    int ret = KTridiagCR(a_gbl, 0, n_eqt, 4, n_batch);
    cudaDeviceSynchronize();
    BPLG_CHECK_ERROR(ret, "BPLG_CR")
}

void BPLG_PCR(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    int ret = KTridiagPCR(a_gbl, 0, n_eqt, 4, n_batch);
    cudaDeviceSynchronize();
    BPLG_CHECK_ERROR(ret, "BPLG_PCR")
}
