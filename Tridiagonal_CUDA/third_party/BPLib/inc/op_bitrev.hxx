//- =======================================================================
//+ Bit-reverse operators 
//- =======================================================================

#pragma once
#ifndef _OP_BITREV
#define _OP_BITREV

//---- Header Dependencies -----------------------------------------------

#include "inc/complex.hxx"

//---- Basic Bit-Reverse Functions ---------------------------------------

// In-place vector bit-reversal operator
template<int RAD, class DTYPE> inline __device__ void
bitrev(DTYPE* a, int stride = 1);

// In-place vector bit-reversal operator, batch operation
template<int SIZE, int RAD, class DTYPE> inline __device__ void
bitrev(DTYPE* reg, int stride = 1) {
	#pragma unroll
	for(int i = 0; i < SIZE; i += RAD)
		bitrev<RAD>(reg + i, stride);
}

// In-place swap operator
template<class T> inline __device__ void
rev_SWAP(T& v1, T&v2) {
	T tmp = v1;
	v1 = v2;
	v2 = tmp;
}

// Out-of-place vector bit-reversal operator
template<int RAD, class DTYPE> inline __device__ void
bitrev(const DTYPE* dstData, int dstStride, DTYPE* srcData, int srcStride = 1) {
	#pragma unroll
	for(int i = 0; i < RAD; i++)
		dstData[i * dstStride] = srcData[i * srcStride];
	bitrev<RAD>(dstData, dstStride);
}

// Out-of-place vector bit-reversal operator
template<int RAD, class DTYPE> inline __device__ void
bitrev(const DTYPE* dstData, DTYPE* srcData, int srcStride = 1) {
	bitrev<RAD>(dstData, 1, srcData, srcStride);
}


//---- Reverse specializations -------------------------------------------

// For complex values (partial specialization not allowed)

template<> inline __device__ void // For compatibility
bitrev< 0>(COMPLEX* a, int stride) { }

template<> inline __device__ void
bitrev< 1>(COMPLEX* a, int stride) { }

template<> inline __device__ void
bitrev< 2>(COMPLEX* a, int stride) { }

template<> inline __device__ void
bitrev< 4>(COMPLEX* a, int stride) {
	rev_SWAP(a[ 1 * stride], a[ 2 * stride]);
}

template<> inline __device__ void
bitrev< 8>(COMPLEX* a, int stride) {
	rev_SWAP(a[ 1 * stride], a[ 4 * stride]);
	rev_SWAP(a[ 3 * stride], a[ 6 * stride]);
}

template<> inline __device__ void
bitrev<16>(COMPLEX* a, int stride) {
	rev_SWAP(a[ 1 * stride], a[ 8 * stride]);
	rev_SWAP(a[ 2 * stride], a[ 4 * stride]);
	rev_SWAP(a[ 3 * stride], a[12 * stride]);
	rev_SWAP(a[ 5 * stride], a[10 * stride]);
	rev_SWAP(a[ 7 * stride], a[14 * stride]);
	rev_SWAP(a[11 * stride], a[13 * stride]);
}

template<> inline __device__ void
bitrev<32>(COMPLEX* a, int stride) {
	rev_SWAP(a[ 1 * stride], a[16 * stride]);
	rev_SWAP(a[ 2 * stride], a[ 8 * stride]);
	rev_SWAP(a[ 3 * stride], a[24 * stride]);
	rev_SWAP(a[ 5 * stride], a[20 * stride]);
	rev_SWAP(a[ 6 * stride], a[12 * stride]);
	rev_SWAP(a[ 7 * stride], a[28 * stride]);
	rev_SWAP(a[ 9 * stride], a[18 * stride]);
	rev_SWAP(a[11 * stride], a[26 * stride]);
	rev_SWAP(a[13 * stride], a[22 * stride]);
	rev_SWAP(a[15 * stride], a[30 * stride]);
	rev_SWAP(a[19 * stride], a[25 * stride]);
	rev_SWAP(a[23 * stride], a[29 * stride]);
}

// For float values (partial specialization not allowed)

template<> inline __device__ void // For compatibility
bitrev< 0>(float* a, int stride) { }

template<> inline __device__ void
bitrev< 1>(float* a, int stride) { }

template<> inline __device__ void
bitrev< 2>(float* a, int stride) { }

template<> inline __device__ void
bitrev< 4>(float* a, int stride) {
	rev_SWAP(a[ 1 * stride], a[ 2 * stride]);
}

template<> inline __device__ void
bitrev< 8>(float* a, int stride) {
	rev_SWAP(a[ 1 * stride], a[ 4 * stride]);
	rev_SWAP(a[ 3 * stride], a[ 6 * stride]);
}

template<> inline __device__ void
bitrev<16>(float* a, int stride) {
	rev_SWAP(a[ 1 * stride], a[ 8 * stride]);
	rev_SWAP(a[ 2 * stride], a[ 4 * stride]);
	rev_SWAP(a[ 3 * stride], a[12 * stride]);
	rev_SWAP(a[ 5 * stride], a[10 * stride]);
	rev_SWAP(a[ 7 * stride], a[14 * stride]);
	rev_SWAP(a[11 * stride], a[13 * stride]);
}

template<> inline __device__ void
bitrev<32>(float* a, int stride) {
	rev_SWAP(a[ 1 * stride], a[16 * stride]);
	rev_SWAP(a[ 2 * stride], a[ 8 * stride]);
	rev_SWAP(a[ 3 * stride], a[24 * stride]);
	rev_SWAP(a[ 5 * stride], a[20 * stride]);
	rev_SWAP(a[ 6 * stride], a[12 * stride]);
	rev_SWAP(a[ 7 * stride], a[28 * stride]);
	rev_SWAP(a[ 9 * stride], a[18 * stride]);
	rev_SWAP(a[11 * stride], a[26 * stride]);
	rev_SWAP(a[13 * stride], a[22 * stride]);
	rev_SWAP(a[15 * stride], a[30 * stride]);
	rev_SWAP(a[19 * stride], a[25 * stride]);
	rev_SWAP(a[23 * stride], a[29 * stride]);
}

#endif // _OP_BITREV


