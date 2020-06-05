//- =======================================================================
//+ Twiddle operator v2.0
//- =======================================================================

#pragma once
#ifndef _OP_TWIDDLE
#define _OP_TWIDDLE

//---- Header Dependencies -----------------------------------------------

#include "inc/complex.hxx"
#include "inc/op_bitrev.hxx"

//---- Compile-time Log2 operator ----------------------------------------

//- First it is defined by induction LOG2<N> = 1 + LOG2< N/2 >
template<int N> struct LOG2 { enum { val = 1 + LOG2< (N>>1) >::val }; };

//- Following the base case is defined, LOG2<1> = 0
template<> struct LOG2<1> { enum { val = 0 }; };

//- This case is defined for compatibility
template<> struct LOG2<0> {	enum { val = 0 }; };

//---- Compile-time Pow operator ----------------------------------------

template<int VAL, int EXP> struct POW {
    enum { val = VAL * POW<VAL, EXP-1>::val };
};

template<int VAL>
struct POW<VAL, 0> { enum{ val = 1 }; };

//---- Fast simultaneous cos and sin -------------------------------------

// Al compilar con '-use_fast_math' se cambia por '__sincosf'
inline __device__ void
fastCosSin(const float angle, COMPLEX &result) {
	sincosf(angle, &result.y, &result.x);
}

/*
// Parece perjudicar la precision, ANG360 seria -2
inline __device__ void
fastCosSinPi(const float angle, COMPLEX &result) {
	sincospif(angle, &result.y, &result.x);
}
*/


//---- Computes the angle for twiddle ------------------------------------
// DIR: Transform direction (compile-time)
// PASS: Accumulated radix passes
// STEP: Next radix-pass
template<int DIR, int PASS, int STEP> inline __device__ float
getAngle(int posId) {
	// Factor de PI para la etapa actual, (compile-time)
	const float angle360 = -6.283185307179586476925286766559;
	const float piFactor = DIR * (angle360 / (PASS * STEP));

	// Mascara para posicion en funcion del numero de etapa (compile-time)
	// Ejemplo: 3 etapas Rad2 -> 00000111, 2 etapas Rad3 -> 00111111
	const int piMask = PASS - 1;

	// Factor efectivo calculado a partir de la posicion *REAL*
	// Para transformadas multi-kernel tener en cuenta el grupo 'blockId'
	return piFactor * (float)(posId & piMask);
}

// Runtime pass
template<int DIR, int STEP> inline __device__ float
getAngle(int pass, int posId) {
	const float ANG360 = -6.283185307179586476925286766559;
	const float piFactor = DIR * ANG360 / (STEP * pass);
	return piFactor * posId;
}

//---- Applies the given angle as twiddle --------------------------------

// Twiddle for a single vector
template<int RAD> inline __device__ void
twiddle(COMPLEX* ptr, float angle, int stride = 1) {
	#pragma unroll
	for(int i = 1; i < RAD; i++) {
		COMPLEX twiddle;
		fastCosSin((float)i * angle, twiddle);
        ptr[i * stride] = ptr[i * stride] * twiddle;
	}
}

// Twiddle for a batch of vectors
template<int SIZE, int RADIX> inline __device__ void
twiddle(COMPLEX* ptr, float angle, int stride = 1) {
	#pragma unroll
	for(int i = 0; i < SIZE; i+= RADIX)
		twiddle<RADIX>(ptr + i, angle, stride);
}


// Template specialization for Radix-2
template<> inline __device__ void
twiddle<2>(COMPLEX* ptr, float angle, int stride) {
	COMPLEX twiddle;
	fastCosSin(angle, twiddle);
	ptr[stride] = ptr[stride] * twiddle;
}

// Template specialization for Radix-1
template<> inline __device__ void
twiddle<1>(COMPLEX* ptr, float angle, int stride) { }

#endif // _OP_TWIDDLE

