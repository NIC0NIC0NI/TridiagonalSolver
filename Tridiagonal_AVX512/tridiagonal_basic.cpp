#include <mkl_lapacke.h>
#include <mkl_lapack.h>
#include "tridiagonal.h"
#include "threading.h"

template<typename val_t>
void Thomas(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, int n_eqt, int n_batch) {
    PARALLEL_FOR_BEGIN(j, 0, n_batch)
        val_t *a = a_gbl + j*n_eqt;
        val_t *b = b_gbl + j*n_eqt;
        val_t *c = c_gbl + j*n_eqt;
        val_t *d = d_gbl + j*n_eqt;
        int i;

        c[0] = c[0]/b[0];
        d[0] = d[0]/b[0];
        for(i = 1; i < n_eqt; i ++) {
            val_t k = (b[i] - c[i-1]*a[i]);
            c[i] = c[i] / k;
            d[i] = (d[i] - d[i-1]*a[i]) / k;
        }
        for(i = n_eqt-2; i >= 0; i --) {
            d[i] = d[i] - c[i]*d[i+1];
        }
    PARALLEL_FOR_END
}


template<typename val_t, int CR_levels>
void CR(val_t *a_gbl, val_t *b_gbl, val_t *c_gbl, val_t *d_gbl, val_t *buffer, int n_eqt, int n_batch) {
    PARALLEL_FOR_BEGIN(j, 0, n_batch)
        val_t *a1 = a_gbl + j*n_eqt;
        val_t *b1 = b_gbl + j*n_eqt;
        val_t *c1 = c_gbl + j*n_eqt;
        val_t *d1 = d_gbl + j*n_eqt;
        val_t *a2 = buffer + 0*n_batch*n_eqt + j*n_eqt;
        val_t *b2 = buffer + 1*n_batch*n_eqt + j*n_eqt;
        val_t *c2 = buffer + 2*n_batch*n_eqt + j*n_eqt;
        val_t *d2 = buffer + 3*n_batch*n_eqt + j*n_eqt;

        int n = n_eqt;
        for(int t = 0; t < CR_levels; t ++)
        {
            n = n / 2;
            #pragma omp simd
            for(int i = 0; i < n; i ++)
            {
                val_t an=0,ap=0,bn=1,bp=1,cn=0,cp=0,dn=0,dp=0;
                if(i > 0)
                {
                    an= a1[2*i-1];
                    bn= b1[2*i-1];
                    cn= c1[2*i-1];
                    dn= d1[2*i-1];
                }
                {
                    ap= a1[2*i+1];
                    bp= b1[2*i+1];
                    cp= c1[2*i+1];
                    dp= d1[2*i+1];
                }
                val_t k1 = a1[2*i]/bn;
                val_t k2 = c1[2*i]/bp;
                a2[i] = -an*k1;
                b2[i] = b1[2*i] - cn*k1 - ap*k2;
                c2[i] = -cp*k2;
                d2[i] = d1[2*i] - dn*k1 - dp*k2;
            }
            a1 = a2;
            b1 = b2;
            c1 = c2;
            d1 = d2;
            a2 = a1 + n;
            b2 = b1 + n;
            c2 = c1 + n;
            d2 = d1 + n;
        }
        {
            c1[0] = c1[0]/b1[0];
            d1[0] = d1[0]/b1[0];
            int i;
            for(i = 1; i < n; i ++)
            {
                val_t k = 1.0 / (b1[i] - c1[i-1]*a1[i]);
                c1[i] = k * c1[i];
                d1[i] = k * (d1[i] - d1[i-1]*a1[i]);
            }
            i = n-1;
            for(i = n-2; i >= 0; i --)
            {
                d1[i] = d1[i] - c1[i]*d1[i+1];
            }
        }
        d2 = d1;
        a1 = a1 - 2*n;
        b1 = b1 - 2*n;
        c1 = c1 - 2*n;
        d1 = d1 - 2*n;
        for(int t = 0; t < CR_levels-1; t ++)
        {
            #pragma omp simd
            for(int i = 0; i < n; i ++)
            {
                val_t ll = 0;
                if(i < n-1) ll = d2[i+1];
                d1[2*i] = d2[i];
                d1[2*i+1] = (d1[2*i+1] - a1[2*i+1]*d2[i] - c1[2*i+1]*ll)/b1[2*i+1];
            }
            n = n * 2;
            d2 = d1;
            a1 = a1 - 2*n;
            b1 = b1 - 2*n;
            c1 = c1 - 2*n;
            d1 = d1 - 2*n;
        }
        a1 = a_gbl + j*n_eqt;
        b1 = b_gbl + j*n_eqt;
        c1 = c_gbl + j*n_eqt;
        d1 = d_gbl + j*n_eqt;
        #pragma omp simd
        for(int i = 0; i < n; i ++)
        {
            val_t ll = 0;
            if(i < n-1) ll = d2[i+1];
            d1[2*i] = d2[i];
            d1[2*i+1] = (d1[2*i+1] - a1[2*i+1]*d2[i] - c1[2*i+1]*ll)/b1[2*i+1];
        }
    PARALLEL_FOR_END
}

void MKL_single(float *a, float *b, float *c, float *d, float *buff, int n_eqt, int n_batch) {
    PARALLEL_FOR_BEGIN(j, 0, n_batch)
        LAPACKE_sgtsv(LAPACK_COL_MAJOR, n_eqt, 1, a+j*n_eqt+1, \
                      b+j*n_eqt, c+j*n_eqt, d+j*n_eqt, n_eqt);
    PARALLEL_FOR_END
}

void MKL_double(double *a, double *b, double *c, double *d, double *buff, int n_eqt, int n_batch) {
    PARALLEL_FOR_BEGIN(j, 0, n_batch)
        LAPACKE_dgtsv(LAPACK_COL_MAJOR, n_eqt, 1, a+j*n_eqt+1, \
                      b+j*n_eqt, c+j*n_eqt, d+j*n_eqt, n_eqt);
    PARALLEL_FOR_END
}

void Thomas_single(float *a, float *b, float *c, float *d, float *buff, int n_eqt, int n_batch) {
    Thomas<float>(a, b, c, d, n_eqt, n_batch);
}

void Thomas_double(double *a, double *b, double *c, double *d, double *buff, int n_eqt, int n_batch) {
    Thomas<double>(a, b, c, d, n_eqt, n_batch);
}

void CR_single(float *a, float *b, float *c, float *d, float *buff, int n_eqt, int n_batch) {
    CR<float, 7>(a, b, c, d, buff, n_eqt, n_batch);
}

void CR_double(double *a, double *b, double *c, double *d, double *buff, int n_eqt, int n_batch) {
    CR<double, 7>(a, b, c, d, buff, n_eqt, n_batch);
}
