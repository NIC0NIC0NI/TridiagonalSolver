//- =======================================================================
//+ Load 
//- =======================================================================

#pragma once
#ifndef _OP_LOAD
#define _OP_LOAD


#include "inc/op_reduce.hxx"


//---- Global Memory Load ------------------------------------------------

//- Template to obtain the corresponding simd vector for given type
template<class DTYPE, int SIZE> struct simd;
template<> struct simd<float, 1> { typedef float   type; };
template<> struct simd<float, 2> { typedef float2  type; };
template<> struct simd<float, 3> { typedef float3  type; };
template<> struct simd<float, 4> { typedef float4  type; };

template<> struct simd<int,2>{typedef int2 type;};
template<> struct simd<int,4>{typedef int4 type;};



//- Packed data load (prevents ShMem exchange before first radix)
template<class DTYPE, int SIZE> struct load {

	//- General version signature
	static inline __device__ void
	col(Eqn<DTYPE>* __restrict__ dst, const char row,
		const DTYPE* __restrict__ src);
	/*{ //! Delete: Generic version is inefficient and not really needed
		#pragma unroll
		for(int i = 0; i < SIZE; i++) {
			DTYPE t = src[i];
			switch(row) {
				case 'x': dst[i].x = t; break;
				case 'y': dst[i].y = t; break;
				case 'z': dst[i].z = t; break;
				case 'w': dst[i].w = t; break;
			}
		}
	}*/
};



template<class DTYPE> struct load<DTYPE, 1> {
	//- Specialization to load 2 consecutive elements using float2
	static inline __device__ void
	col(Eqn<DTYPE>* __restrict__ dst, const char row,
		const DTYPE* __restrict__ src)
	{
		typename simd<DTYPE, 1>::type t = *(typename simd<DTYPE, 1>::type*)(src);
		switch(row) {
			case 'x': (*dst).x = t; break;
			case 'y': (*dst).y = t; break;
			case 'z': (*dst).z = t; break;
			case 'w': (*dst).w = t; break;
		}
	}
};
template<class DTYPE> struct load<DTYPE, 2> {
	//- Specialization to load 2 consecutive elements using float2
	static inline __device__ void
	col(Eqn<DTYPE>* __restrict__ dst, const char row,
		const DTYPE* __restrict__ src)
	{
		typename simd<DTYPE, 2>::type t = *(typename simd<DTYPE, 2>::type*)(src);
		switch(row) {
			case 'x': dst[0].x = t.x; dst[1].x = t.y; break;
			case 'y': dst[0].y = t.x; dst[1].y = t.y; break;
			case 'z': dst[0].z = t.x; dst[1].z = t.y; break;
			case 'w': dst[0].w = t.x; dst[1].w = t.y; break;
		}
	}
};

template<class DTYPE> struct load<DTYPE, 3> {
	//- Specialization to load 2 consecutive elements using float2
	static inline __device__ void
	col(Eqn<DTYPE>* __restrict__ dst, const char row,
		const DTYPE* __restrict__ src)
	{
		typename simd<DTYPE, 3>::type t = *(typename simd<DTYPE, 3>::type*)(src);
		switch(row) {
			case 'x': dst[0].x = t.x; dst[1].x = t.y; dst[2].x=t.z; break;
			case 'y': dst[0].y = t.x; dst[1].y = t.y; dst[2].y=t.z; break;
			case 'z': dst[0].z = t.x; dst[1].z = t.y; dst[2].z=t.z; break;
			case 'w': dst[0].w = t.x; dst[1].w = t.y; dst[2].w=t.z; break;
		}
	}
};

template<class DTYPE> struct load<DTYPE, 4> {
	//- Specialization to load 4 consecutive elements using float4
	static inline __device__ void
		col(Eqn<DTYPE>* __restrict__ dst, const char row,
			const DTYPE* __restrict__ src)
	{
		typename simd<DTYPE, 4>::type t = *(typename simd<DTYPE, 4>::type*)(src);
		switch(row) {
			case 'x': dst[0].x = t.x; dst[1].x = t.y;
					  dst[2].x = t.z; dst[3].x = t.w; break;
			case 'y': dst[0].y = t.x; dst[1].y = t.y;
					  dst[2].y = t.z; dst[3].y = t.w; break;
			case 'z': dst[0].z = t.x; dst[1].z = t.y;
					  dst[2].z = t.z; dst[3].z = t.w; break;
			case 'w': dst[0].w = t.x; dst[1].w = t.y;
					  dst[2].w = t.z; dst[3].w = t.w; break;
		}
	}
};
	
	
template<class DTYPE> struct load<DTYPE, 8> {
	//- Specialization to load 8 consecutive elements, uses 2xfloat4
	static inline __device__ void
		col(Eqn<DTYPE>* __restrict__ dst, const char row,
		const DTYPE* __restrict__ src)
	{
		// Vector8 loads are emulated with two Vector4 loads
		load<DTYPE,4>::col(dst  , row, src  );
		load<DTYPE,4>::col(dst+4, row, src+4);
	}

};

#endif // _OP_LOAD

