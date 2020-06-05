//- =======================================================================
//+ Scale operator v2.0
//- =======================================================================

#pragma once
#ifndef _OP_SCALE
#define _OP_SCALE

//---- Header Dependencies -----------------------------------------------

#include "inc/complex.hxx"

//---- Basic Copy Functions ----------------------------------------------

// Scales a vector of signals by a predefined value
template<int SIZE, int N, class DTYPE> inline __device__ void
scale(DTYPE* a, int stride = 1) {
	const float scale = 1.0f / (float)N; // Compile-time comst
	#pragma unroll
	for(int i = 0; i < SIZE; i++)
		a[i * stride] = scale * a[i * stride];
}

// Scales a vector of signals by the specified value
template<int SIZE, int N, class DTYPE> inline __device__ void
scale(DTYPE* a, int stride, float scale) {
	#pragma unroll
	for(int i = 0; i < SIZE; i++)
		a[i * stride] = scale * a[i * stride];
}

// Scales a vector by the specified value, no stride
template<int SIZE, int N, class DTYPE> inline __device__ void
scale(DTYPE* a, float scale) {
	scale<SIZE, N>(a, 1, scale);
}

// Scales a single value by a predefined value
template<int N, class DTYPE> inline __device__ DTYPE
scale(DTYPE val) { return (1.0f / N) * val; }

// Scales a single signal by a predefined value
template<int N, class DTYPE> inline __device__ void
scale(DTYPE* val) { scale<N, N>(val); }

#endif // _OP_SCALE

