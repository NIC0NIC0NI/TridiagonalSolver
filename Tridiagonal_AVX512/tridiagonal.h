#ifndef TRIDIAGONAL_H_INCLUDED
#define TRIDIAGONAL_H_INCLUDED 1

#ifdef __cplusplus
extern "C" {  /* callable from C */
#endif

int check_single(const float *a, const float *b, int size, float *err);
int check_double(const double *a, const double *b, int size, double *err);

void MKL_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, float *buf, int n_eqt, int n_batch);
void Thomas_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, float *buf, int n_eqt, int n_batch);
void CR_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, float *buf, int n_eqt, int n_batch);
void WM4_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, float *buf, int n_eqt, int n_batch);
void WM3_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, float *buf, int n_eqt, int n_batch);
void PCR4_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, float *buf, int n_eqt, int n_batch);
void PCR3_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, float *buf, int n_eqt, int n_batch);

void MKL_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, double *buff, int n_eqt, int n_batch);
void Thomas_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, double *buff, int n_eqt, int n_batch);
void CR_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, double *buff, int n_eqt, int n_batch);
void WM3_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, double *buff, int n_eqt, int n_batch);
void WM2_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, double *buff, int n_eqt, int n_batch);
void PCR3_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, double *buff, int n_eqt, int n_batch);
void PCR2_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, double *buff, int n_eqt, int n_batch);

#ifdef __cplusplus
}
#endif

#endif