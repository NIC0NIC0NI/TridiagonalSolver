/**
 * Check the error of the result
 * Use pairwise sum for more accurate sum (1-norm and 2-norm)
 * Treat NaN specially for maximum value (infinity-norm)
 */

#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include "tridiagonal.h"

#define STACK_SIZE 32

#define MAX_SINGLE(vec1, vec2, ii1, ii2) {\
    __mmask16 kk = _mm512_kor(_mm512_cmple_ps_mask(vec1, vec2), _mm512_cmpunord_ps_mask(vec2, vec2));  \
    vec1 = _mm512_mask_mov_ps(vec1, kk, vec2); \
    ii1 = _mm512_mask_mov_epi32(ii1, kk, ii2); \
}

#define MAX_DOUBLE(vec1, vec2, ii1, ii2) {\
    __mmask8 kk = _mm512_kor(_mm512_cmple_pd_mask(vec1, vec2), _mm512_cmpunord_pd_mask(vec2, vec2)); \
    vec1 = _mm512_mask_mov_pd(vec1, kk, vec2); \
    ii1 = _mm512_mask_mov_epi64(ii1, kk, ii2); \
}

#define REDUCE_MAX_SINGLE(xx, vec, ii, vii) { \
    __mmask16 kk = _mm512_cmpunord_ps_mask(vec, vec); \
    float pnan = _mm512_reduce_add_ps(_mm512_maskz_mov_ps(kk, vec)); \
    if(isnan(pnan)) { \
        xx = pnan; \
        ii = _mm512_reduce_max_epi32(_mm512_maskz_mov_epi32(kk, vii)); \
    } else { \
        xx = _mm512_reduce_max_ps(vec); \
        ii = _mm512_reduce_max_epi32(_mm512_maskz_mov_epi32( \
        _mm512_cmpeq_ps_mask(vec, _mm512_set1_ps(xx)), vii)); \
    } \
}

#define REDUCE_MAX_DOUBLE(xx, vec, ii, vii) { \
    __mmask8 kk = _mm512_cmpunord_pd_mask(vec, vec); \
    double pnan = _mm512_reduce_add_pd(_mm512_maskz_mov_pd(kk, vec)); \
    if(isnan(pnan)) { \
        xx = pnan; \
        ii = _mm512_reduce_max_epi64(_mm512_maskz_mov_epi64(kk, vii)); \
    } else { \
        xx = _mm512_reduce_max_pd(vec); \
        ii = _mm512_reduce_max_epi64(_mm512_maskz_mov_epi64( \
        _mm512_cmpeq_pd_mask(vec, _mm512_set1_pd(xx)), vii)); \
    } \
}

int check_single(const float *a, const float *b, int size, float *err) {
    const __m512i delta = _mm512_set1_epi32(16);
    __m512i index = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0), mi;
    __m512 stack1[STACK_SIZE], stack2[STACK_SIZE];
    __m512 sum = _mm512_setzero_ps(), sqsum = _mm512_setzero_ps(), m = _mm512_setzero_ps();
    int i, bit, top = 0;
    for(i = 0; i < size - 63; i += 64) {   /* pairwise sum */
        __m512 err0 = _mm512_abs_ps(_mm512_sub_ps(_mm512_load_ps(a + i     ), _mm512_load_ps(b + i     )));
        __m512 err1 = _mm512_abs_ps(_mm512_sub_ps(_mm512_load_ps(a + i + 16), _mm512_load_ps(b + i + 16)));
        __m512 err2 = _mm512_abs_ps(_mm512_sub_ps(_mm512_load_ps(a + i + 32), _mm512_load_ps(b + i + 32)));
        __m512 err3 = _mm512_abs_ps(_mm512_sub_ps(_mm512_load_ps(a + i + 48), _mm512_load_ps(b + i + 48)));

        __m512 w1 = _mm512_add_ps(err0, err1);
        __m512 u1 = _mm512_add_ps(err2, err3);
        __m512 w2 = _mm512_add_ps(_mm512_mul_ps(err0, err0), _mm512_mul_ps(err1, err1));
        __m512 u2 = _mm512_add_ps(_mm512_mul_ps(err2, err2), _mm512_mul_ps(err3, err3));
        __m512 v1 = _mm512_add_ps(w1, u1);
        __m512 v2 = _mm512_add_ps(w2, u2);
        for(bit = 64; (i & bit) != 0; bit = bit << 1) {
            v1 = _mm512_add_ps(v1, stack1[--top]);
            v2 = _mm512_add_ps(v2, stack2[top]);
        }
        stack1[top] = v1;
        stack2[top++] = v2;

        __m512i i0 = index;
        __m512i i1 = _mm512_add_epi32(i0, delta);
        __m512i i2 = _mm512_add_epi32(i1, delta);
        __m512i i3 = _mm512_add_epi32(i2, delta);
        MAX_SINGLE(err0, err1, i0, i1)
        MAX_SINGLE(err2, err3, i2, i3)
        MAX_SINGLE(err0, err2, i0, i2)
        MAX_SINGLE(m, err0, mi, i0)
        index = _mm512_add_epi32(i3, delta);
    }
    for(; i < size; i += 16) {
        __m512 err0 = _mm512_abs_ps(_mm512_sub_ps(_mm512_load_ps(a + i), _mm512_load_ps(b + i)));
        sum = _mm512_add_ps(sum, err0);
        sqsum = _mm512_add_ps(sqsum, _mm512_mul_ps(err0, err0));
        __m512i i0 = index;
        MAX_SINGLE(m, err0, mi, i0)
        index = _mm512_add_epi32(i0, delta);
    }
    for(i = top - 1; i >= 0; --i) {
        sum = _mm512_add_ps(sum, stack1[i]);
        sqsum = _mm512_add_ps(sqsum, stack2[i]);
    }
    float mm;
    int mmi;
    REDUCE_MAX_SINGLE(mm, m, mmi, mi)
    err[0] = _mm512_reduce_add_ps(sum) / size;
    err[1] = sqrtf(_mm512_reduce_add_ps(sqsum) / size);
    err[2] = mm;
    return mmi;
}


int check_double(const double *a, const double *b, int size, double *err) {
    const __m512i delta = _mm512_set1_epi64(8);
    __m512i index = _mm512_set_epi64(7,6,5,4,3,2,1,0), mi;
    __m512d stack1[STACK_SIZE], stack2[STACK_SIZE];
    __m512d sum = _mm512_setzero_pd(), sqsum = _mm512_setzero_pd(), m = _mm512_setzero_pd();
    int i, bit, top = 0;
    for(i = 0; i < size - 31; i += 32) {   /* pairwise sum */
        __m512d err0 = _mm512_abs_pd(_mm512_sub_pd(_mm512_load_pd(a + i     ), _mm512_load_pd(b + i     )));
        __m512d err1 = _mm512_abs_pd(_mm512_sub_pd(_mm512_load_pd(a + i + 8 ), _mm512_load_pd(b + i + 8 )));
        __m512d err2 = _mm512_abs_pd(_mm512_sub_pd(_mm512_load_pd(a + i + 16), _mm512_load_pd(b + i + 16)));
        __m512d err3 = _mm512_abs_pd(_mm512_sub_pd(_mm512_load_pd(a + i + 24), _mm512_load_pd(b + i + 24)));

        __m512d w1 = _mm512_add_pd(err0, err1);
        __m512d u1 = _mm512_add_pd(err2, err3);
        __m512d w2 = _mm512_add_pd(_mm512_mul_pd(err0, err0), _mm512_mul_pd(err1, err1));
        __m512d u2 = _mm512_add_pd(_mm512_mul_pd(err2, err2), _mm512_mul_pd(err3, err3));
        __m512d v1 = _mm512_add_pd(w1, u1);
        __m512d v2 = _mm512_add_pd(w2, u2);
        for(bit = 32; (i & bit) != 0; bit = bit << 1) {
            v1 = _mm512_add_pd(v1, stack1[--top]);
            v2 = _mm512_add_pd(v2, stack2[top]);
        }
        stack1[top] = v1;
        stack2[top++] = v2;

        __m512i i0 = index;
        __m512i i1 = _mm512_add_epi64(i0, delta);
        __m512i i2 = _mm512_add_epi64(i1, delta);
        __m512i i3 = _mm512_add_epi64(i2, delta);
        MAX_DOUBLE(err0, err1, i0, i1)
        MAX_DOUBLE(err2, err3, i2, i3)
        MAX_DOUBLE(err0, err2, i0, i2)
        MAX_DOUBLE(m, err0, mi, i0)
        index = _mm512_add_epi64(i3, delta);
    }
    for(; i < size; i += 8) {
        __m512d err0 = _mm512_abs_pd(_mm512_sub_pd(_mm512_load_pd(a + i), _mm512_load_pd(b + i)));
        sum = _mm512_add_pd(sum, err0);
        sqsum = _mm512_add_pd(sqsum, _mm512_mul_pd(err0, err0));
        __m512i i0 = index;
        MAX_DOUBLE(m, err0, mi, i0)
        index = _mm512_add_epi64(i0, delta);
    }
    for(i = top - 1; i >= 0; --i) {
        sum = _mm512_add_pd(sum, stack1[i]);
        sqsum = _mm512_add_pd(sqsum, stack2[i]);
    }
    float mm;
    int mmi;
    REDUCE_MAX_DOUBLE(mm, m, mmi, mi)
    err[0] = _mm512_reduce_add_pd(sum) / size;
    err[1] = sqrt(_mm512_reduce_add_pd(sqsum) / size);
    err[2] = mm;
    return mmi;
}
