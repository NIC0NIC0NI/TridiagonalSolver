//- =======================================================================
//+ Radix operator v2.0
//- =======================================================================

#pragma once
#ifndef _OP_RADIX
#define _OP_RADIX

//---- Header Dependencies -----------------------------------------------

#include "inc/complex.hxx"
#include "inc/op_copy.hxx"
#include "inc/op_bitrev.hxx"
#include "inc/op_twiddle.hxx"
#include "inc/op_scale.hxx"
#include "inc/op_bfly.hxx"

//- Single vector radix
template<int RAD, int DIR> inline __device__ void
radix(COMPLEX* data, int stride = 1) {
	butterfly<RAD, DIR>(data, stride);
	bitrev<RAD>(data, stride);
}

//- Versiones optimizadas para el caso nulo radix-1
template<> subkernel void radix< 1, 1>(COMPLEX* data, int stride) { }
template<> subkernel void radix< 1,-1>(COMPLEX* data, int stride) { }

//- Single vector radix with angle
template<int RAD, int DIR> inline __device__ void
radix(COMPLEX* data, float ang, int stride = 1) {
	if(DIR!= 0) twiddle<RAD>(data, ang, stride);
	butterfly<RAD, DIR>(data, stride);
	bitrev<RAD>(data, stride);
}

//- Versiones optimizadas para el caso nulo radix-1
template<> subkernel void radix< 1, 1>(COMPLEX* data, float ang, int stride) { }
template<> subkernel void radix< 1,-1>(COMPLEX* data, float ang, int stride) { }

// ---- Batch version ----------------------------------------------------

//- Batch vector radix with angle (stride not supported)
template<int SIZE, int RAD, int DIR> inline __device__ void
radix(COMPLEX* data, float ang) {
	#pragma unroll
	for(int i = 0; i < SIZE; i += RAD)
		radix<RAD, DIR>(data + i, ang);
}

//- Batch vector radix (BETA compile-time stride support)
template<int SIZE, int RAD, int DIR> inline __device__ void
radix(COMPLEX* data, int stride = 1) {
	if(stride == 1) {
		#pragma unroll
		for(int i = 0; i < SIZE; i += RAD)
			radix<RAD, DIR>(data + i);
		return;
	}
	if(stride == SIZE/RAD) {
		#pragma unroll
		for(int i = 0; i < SIZE/RAD; i++)
			radix<RAD, DIR>(data + i, SIZE/RAD);
		return;
	}
}

//- Batch vector radix (BETA compile-time stride support)
template<int SIZE, int RAD, int DIR> inline __device__ void
mradix(COMPLEX* data, float ang) {
	#pragma unroll
	for(int i = 0; i < SIZE/RAD; i++)
		radix<RAD, DIR>(data + i, ang, SIZE/RAD);
	return;
}

// ---- Special operators ------------------------------------------------

//- Compile-time mixed-Radix calculator (BETA)
template<int N, int R> struct MIXR { enum { val = MIXR<N/R, R>::val }; };
template<> struct MIXR< 1, 2> { enum { val = 2 }; };
template<> struct MIXR< 1, 4> { enum { val = 4 }; };
template<> struct MIXR< 2, 4> { enum { val = 2 }; };
template<> struct MIXR< 1, 8> { enum { val = 8 }; };
template<> struct MIXR< 2, 8> { enum { val = 2 }; };
template<> struct MIXR< 4, 8> { enum { val = 4 }; };
template<> struct MIXR< 1,16> { enum { val =16 }; };
template<> struct MIXR< 2,16> { enum { val = 2 }; };
template<> struct MIXR< 4,16> { enum { val = 4 }; };
template<> struct MIXR< 8,16> { enum { val = 8 }; };

template<> struct MIXR< 1,32> { enum { val =32 }; };
template<> struct MIXR< 2,32> { enum { val = 2 }; };
template<> struct MIXR< 4,32> { enum { val = 4 }; };
template<> struct MIXR< 8,32> { enum { val = 8 }; };
template<> struct MIXR<16,32> { enum { val =16 }; };


//- Radix modificado para transformadas reales
template<int SIZE, int RAD, int DIR> inline __device__ void
rradix(COMPLEX* data, float ang = 1.0f) {
	#pragma unroll
	for(int i = 0; i < SIZE; i += RAD)
		rbutterfly<RAD, DIR>(data + i);
}

#endif // _OP_RADIX

