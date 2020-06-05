
//- =======================================================================
//+ Shuffle Radix
//- =======================================================================

#pragma once
#ifndef _OP_SHFL
#define _OP_SHFL


#include "inc/op_reduce.hxx"

template<class DTYPE> struct shfl {

	//- General version signature
	static inline __device__ Eqn<DTYPE>
	shfl_Eq(const Eqn<DTYPE> src, const int id,
		const int width)
	{
		Eqn<DTYPE> eq;

		return eq;	
	}


	static inline __device__ Eqn<DTYPE>
	shfl_Eq_Up(const Eqn<DTYPE> src, const int id,const int width)
	{
		Eqn<DTYPE> eq;

		return eq;	
	}

	static inline __device__ Eqn<DTYPE>
	shfl_Eq_Down(const Eqn<DTYPE> src, const int id,const int width)
	{
		Eqn<DTYPE> eq;

		return eq;	
	}

	
	static inline __device__ DTYPE
	shflDTYPE (const DTYPE var, const int id,
		const int width)
	{
		DTYPE x;

		return x;
	}


	static inline __device__ DTYPE
	shfl_Up (const DTYPE src, const int id,const int width)
	{
		DTYPE x;
		return x;
	}
	
	
	static inline __device__ DTYPE
	shfl_Down (const DTYPE src, const int id,const int width)
	{
		DTYPE x;
		return x;
	}



};

template<> struct shfl<float> {

	static inline __device__ Eqn<float>
	shfl_Eq(const Eqn<float> src, const int id,
		const int width)
	{	
		float a = __shfl(src.x,id,width);
		float b = __shfl(src.y,id,width);
		float c = __shfl(src.z,id,width);
		float d = __shfl(src.w,id,width);
		//Eqn<float>(); //Constructor de la ecuacion
		return Eqn<float>(a,b,c,d); 	
	}

	static inline __device__ Eqn<float>
	shfl_Eq_Up(const Eqn<float> src, const int id,const int width)
	{
		float a = __shfl_up(src.x,id,width);
		float b = __shfl_up(src.y,id,width);
		float c = __shfl_up(src.z,id,width);
		float d = __shfl_up(src.w,id,width);
		//Eqn<float>(); //Constructor de la ecuacion
		return Eqn<float>(a,b,c,d); 		
	}

	static inline __device__ Eqn<float>
	shfl_Eq_Down(const Eqn<float> src, const int id,const int width)
	{
		float a = __shfl_down(src.x,id,width);
		float b = __shfl_down(src.y,id,width);
		float c = __shfl_down(src.z,id,width);
		float d = __shfl_down(src.w,id,width);
		//Eqn<float>(); //Constructor de la ecuacion
		return Eqn<float>(a,b,c,d); 	
	}

	static inline __device__ float
	shflDTYPE (const float var, const int id,
	 	const int width)
	{
		return __shfl(var,id,width);
	}

	static inline __device__ float
	shfl_Up (const float src, const int id,const int width)
	{
		float x = __shfl_up(src,id,width);
		return x;
	}
	
	
	static inline __device__ float
	shfl_Down (const float src, const int id,const int width)
	{
		float x = __shfl_down (src,id,width);
		return x;
	}


};
#endif // _OP_SHFL
