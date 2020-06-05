
//- =======================================================================
//+ Mix Radix
//- =======================================================================

#pragma once
#ifndef _OP_MIXR
#define _OP_MIXR


//- Compile-time mixed-Radix calculator (BETA)
//template<int N, int R> struct MIXR { enum { val = MIXR<N/R, R>::val }; };

// General case for size N and radix R
template<int N, int R> struct MIXR {
	enum { val = (N >= R) ? MIXR<N/R, R>::val : N}; };

// Base case when the signal cannot be divided
template<int R> struct MIXR< 1, R> { enum { val = R }; };

// Degenerated case, avoids compile errors
template<int R> struct MIXR< 0, R> { enum { val = 0 }; };


__device__ const float epsilon = 1e-5f;

#endif // _OP_MIXR
