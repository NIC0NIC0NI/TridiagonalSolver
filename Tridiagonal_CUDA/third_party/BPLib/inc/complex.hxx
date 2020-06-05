//- =======================================================================
//+ Complex operations
//- =======================================================================

#pragma once
#ifndef _COMPLEX
#define _COMPLEX

//---- Header Definitions ------------------------------------------------

#define COMPLEX float2

//---- Basic Complex Functions -------------------------------------------

inline __device__ COMPLEX
	make_COMPLEX(float x, float y) {
	float2 t; t.x = x; t.y = y; return t;
}

inline __device__ COMPLEX
conj(COMPLEX a) {
	return make_float2(a.x, -a.y);
}

inline __device__ COMPLEX
operator*(COMPLEX a, COMPLEX b) {
	return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

inline __device__ COMPLEX
operator*(COMPLEX a, float b) {
	return make_float2(b* a.x, b* a.y);
}

inline __device__ COMPLEX
operator*(float a, COMPLEX b) {
	return make_float2(a* b.x, a* b.y);
}

inline __device__ void
operator*=(COMPLEX &a, float b) {
    a.x *= b; a.y *= b;
}

inline __device__ void
operator*=(COMPLEX &a, COMPLEX b) {
	COMPLEX ta = a;
	a.x = ta.x*b.x - ta.y*b.y;
	a.y = ta.x*b.y + ta.y*b.x;
	// a = make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

inline __device__ COMPLEX
operator+(COMPLEX a, COMPLEX b) {
	return make_float2(a.x + b.x, a.y + b.y);
}

inline __device__ COMPLEX
operator-(COMPLEX a, COMPLEX b) {
	return make_float2(a.x - b.x, a.y - b.y);
}

inline __device__ COMPLEX
operator-(COMPLEX a) {
	return make_float2(-a.x, -a.y);
}

#endif // _COMPLEX

