#ifndef VEC_H_INCLUDED
#define VEC_H_INCLUDED

#include <immintrin.h>

#if defined(RCP14)
#define RCP14PD
#define RCP14PS
#elif defined(RCP28)
#define RCP28PD
#define PCR28PS
#elif defined(DIV)
#define DIVPD
#define DIVPS
#else
#define RCP28PS
#define DIVPD
#endif

/* Arithmetics */

inline __m512d div(__m512d a, __m512d b) {
#if defined(RCP14PD)
    return _mm512_mul_pd(a, _mm512_rcp14_pd(b));
#elif defined(RCP28PD)
    return _mm512_mul_pd(a, _mm512_rcp28_pd(b));
#elif defined(DIVPD)
    return _mm512_div_pd(a, b);
#endif
}

inline __m512d maskz_div(__mmask8 mask, __m512d a, __m512d b) {
#if defined(RCP14PD)
    return _mm512_maskz_mul_pd(mask, a, _mm512_rcp14_pd(b));
#elif defined(RCP28PD)
    return _mm512_maskz_mul_pd(mask, a, _mm512_rcp28_pd(b));
#elif defined(DIVPD)
    return _mm512_maskz_div_pd(mask, a, b);
#endif
}

inline __m512 div(__m512 a, __m512 b) {
#if defined(RCP14PS)
    return _mm512_mul_ps(a, _mm512_rcp14_ps(b));
#elif defined(RCP28PS)
    return _mm512_mul_ps(a, _mm512_rcp28_ps(b));
#elif defined(DIVPS)
    return _mm512_div_ps(a, b);
#endif
}

inline __m512 maskz_div(__mmask16 mask, __m512 a, __m512 b) {
#if defined(RCP14PS)
    return _mm512_maskz_mul_ps(mask, a, _mm512_rcp14_ps(b));
#elif defined(RCP28PS)
    return _mm512_maskz_mul_ps(mask, a, _mm512_rcp28_ps(b));
#elif defined(DIVPS)
    return _mm512_maskz_div_ps(mask, a, b);
#endif
}


#ifdef NEGXOR
inline __m512 nmul(__m512 a, __m512 b) {
    return _mm512_mul_ps(_mm512_castsi512_ps(_mm512_xor_si512(_mm512_set1_epi32(0x80000000), _mm512_castps_si512(a))), b);
}
inline __m512d nmul(__m512d a, __m512d b) {
    return _mm512_mul_pd(_mm512_castsi512_pd(_mm512_xor_si512(_mm512_set1_epi64(0x8000000000000000ll), _mm512_castpd_si512(a))), b);
}
#else
inline __m512 nmul(__m512 a, __m512 b) {
    return _mm512_fnmadd_ps(a, b, _mm512_setzero_ps());
}
inline __m512d nmul(__m512d a, __m512d b) {
    return _mm512_fnmadd_pd(a, b, _mm512_setzero_pd());
}
#endif

inline __m512 operator +(__m512 a, __m512 b) {
    return _mm512_add_ps(a, b);
}
inline __m512d operator +(__m512d a, __m512d b) {
    return _mm512_add_pd(a, b);
}
inline __m512 operator -(__m512 a, __m512 b) {
    return _mm512_sub_ps(a, b);
}
inline __m512d operator -(__m512d a, __m512d b) {
    return _mm512_sub_pd(a, b);
}
inline __m512 operator *(__m512 a, __m512 b) {
    return _mm512_mul_ps(a, b);
}
inline __m512d operator *(__m512d a, __m512d b) {
    return _mm512_mul_pd(a, b);
}

#endif