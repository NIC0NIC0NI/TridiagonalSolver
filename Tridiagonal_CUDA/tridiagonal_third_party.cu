#include <iostream>
#include <cusparse.h>
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

static cusparseHandle_t cusparse_handle;
static void *buffer;

int cuSparse_init_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    size_t size;
    int ret = cusparseCreate(&cusparse_handle);
    if(ret != CUSPARSE_STATUS_SUCCESS) {
        return ret;
    }
    cusparseSgtsv2StridedBatch_bufferSizeExt(cusparse_handle, n_eqt, a_gbl, b_gbl, c_gbl, d_gbl, n_batch, n_eqt, &size);
    ret = cudaMalloc(&buffer, size);
    if(ret != cudaSuccess) {
        cusparseDestroy(cusparse_handle);
    }
    return ret;
}

int cuSparse_init_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch) {
    size_t size;
    int ret = cusparseCreate(&cusparse_handle);
    if(ret != CUSPARSE_STATUS_SUCCESS) {
        return ret;
    }
    cusparseDgtsv2StridedBatch_bufferSizeExt(cusparse_handle, n_eqt, a_gbl, b_gbl, c_gbl, d_gbl, n_batch, n_eqt, &size);
    ret = cudaMalloc(&buffer, size);
    if(ret != cudaSuccess) {
        cusparseDestroy(cusparse_handle);
    }
    return ret;
}

void cuSparse_final() {
    cudaFree(buffer);
    cusparseDestroy(cusparse_handle);
}

void cuSparse_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch) {
    cusparseSgtsv2StridedBatch(cusparse_handle, n_eqt, a_gbl, b_gbl, c_gbl, d_gbl, n_batch, n_eqt, buffer);
    cudaDeviceSynchronize();
}
void cuSparse_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch) {
    cusparseDgtsv2StridedBatch(cusparse_handle, n_eqt, a_gbl, b_gbl, c_gbl, d_gbl, n_batch, n_eqt, buffer);
    cudaDeviceSynchronize();
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
