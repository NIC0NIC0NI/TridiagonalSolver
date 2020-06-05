#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED 1

#include<cstddef>

void init_batch(const float *a, const float *b, const float *c, const float *d, \
                float *a_bat, float *b_bat, float *c_bat, float *d_bat, int size, int batch, int layout);
void init_batch(const double *a, const double *b, const double *c, const double *d, \
                double *a_bat, double *b_bat, double *c_bat, double *d_bat, int size, int batch, int layout);

int check(float *ans0, const float *ans1, float *buffer0, float *buffer, int *ibuffer, int size, float *stat, int layout);
int check(double *ans0, const double *ans1, double *buffer0, double *buffer, int *ibuffer, int size, double *stat, int layout);

void *cuda_malloc(std::size_t size);
void cuda_free(void *p);

#endif