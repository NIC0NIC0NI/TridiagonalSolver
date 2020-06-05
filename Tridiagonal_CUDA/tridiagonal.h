#ifndef TRIDIAGONAL_H_INCLUDED
#define TRIDIAGONAL_H_INCLUDED 1 

#ifdef __cplusplus
extern "C" {
#endif

int cuSparse_init_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);
int cuSparse_init_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch);
void cuSparse_final();
void cuSparse_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);
void cuSparse_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch);

/* BPLG */
void BPLG_WM(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);
void BPLG_LF(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);
void BPLG_CR(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);
void BPLG_PCR(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);

/* naive Thomas */
void Thomas_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);
void Thomas_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch);

/* naive CR */
int CR_init_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);
int CR_init_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch);
void CR_final();
void CR_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);
void CR_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch);

void WM5_shmem_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);
void WM4_shmem_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);
void PCR5_shmem_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);
void PCR4_shmem_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);

void WM5_reg_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);
void WM4_reg_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);
void PCR5_reg_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);
void PCR4_reg_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);

void WM5_shmem_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch);
void WM4_shmem_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch);
void PCR5_shmem_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch);
void PCR4_shmem_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch);

void WM5_reg_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch);
void WM4_reg_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch);
void PCR5_reg_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch);
void PCR4_reg_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch);

/* reg-packed CR */
void CR_reg4_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch);
void CR_reg8_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch);
void CR_reg16_double(double *a_gbl, double *b_gbl, double *c_gbl, double *d_gbl, int n_eqt, int n_batch);
void CR_reg4_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);
void CR_reg8_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);
void CR_reg16_single(float *a_gbl, float *b_gbl, float *c_gbl, float *d_gbl, int n_eqt, int n_batch);

#ifdef __cplusplus
}
#endif

#endif