//- =======================================================================
//+ Copy operator 
//- =======================================================================

#pragma once
#ifndef _OP_COPY
#define _OP_COPY

//---- Header Dependencies -----------------------------------------------

#include "inc/complex.hxx"
#pragma warning(disable : 4068)  // No avisa al encontrar pragma unroll

//---- Basic Copy Functions ----------------------------------------------
//- Asignment operator must be supported by the template type

template<int N, class DTYPE> inline __device__ void
copy(      DTYPE* dstData, const int dstStride,
     const DTYPE* srcData, const int srcStride = 1) {
	#pragma unroll
    for(int i = 0; i < N; i++)
        dstData[i * dstStride] = srcData[i * srcStride];
}

template<int N, class DTYPE> inline __device__ void
copy(      DTYPE* dstData,
     const DTYPE* srcData, const int srcStride = 1) {
	#pragma unroll
    for(int i = 0; i < N; i++)
        dstData[i] = srcData[i * srcStride];
}





template<int N, int butterfly, class DTYPE> __inline__ __device__ void
copy(      DTYPE* __restrict__ dstData,
     const DTYPE* __restrict__ srcData, const int srcStride, const int offset) {
	

    
    for(int i = 0; i < (N/butterfly); i++)        
	for(int j=0;j<butterfly ; j++ )
		dstData[i*butterfly + j] = srcData[j * srcStride + i * offset];


}


template<int N, class DTYPE> __inline__ __device__ void
copy(      DTYPE* __restrict__ dstData,
     const DTYPE* __restrict__ srcData, const int srcStride, const int condition) {
	
#pragma unroll
    for(int i = 0; i < N-1; i++)
        dstData[i] = srcData[i * srcStride];

    if(condition)
	dstData[N-1]=srcData[(N-2)*srcStride];
    else
	dstData[N-1]=srcData[(N-1)*srcStride];	

}

template<int N, class DTYPE> __inline__ __device__ void
copy2(      DTYPE* __restrict__ dstData,
     const DTYPE* __restrict__ srcData, const bool srcStride, const int treehold) {
    
	if(( (threadIdx.x>=treehold)*srcStride)||((threadIdx.x<(blockDim.x-treehold))*(!srcStride) ) ){
        	dstData[0] = srcData[0];
}
	

}



template<int N, const bool srcStride, const int RAD, class DTYPE> __inline__ __device__ void
copy(      DTYPE* __restrict__ dstData,
     const DTYPE* __restrict__ srcData, const int treehold) {
    
	if(( (threadIdx.x>=treehold)*srcStride)||((threadIdx.x<(RAD*blockDim.x-treehold))*(!srcStride) ) ){
        	dstData[0] = srcData[0];
}
	

}



// Only for cap 3.0 or higher
// Make sure there are no race conditions
// a) Different read and write memory buffers
// b) Same buffer, but with same data access pattern and a sync
template<int N, class DTYPE> inline __device__ void
copyLDG(   DTYPE* dstData, 
     const DTYPE* srcData, const int srcStride = 1) {
	#pragma unroll
    for(int i = 0; i < N; i++)
        dstData[i] = srcData[i * srcStride];
        //dstData[i] = __ldg(srcData + i * srcStride);
}

/*
template<int N, class DTYPE> inline __device__ void
copy(DTYPE* dstData, int dstStride, const DTYPE* srcData) {
	#pragma unroll
	for(int i = 0; i < N; i++) 
		dstData[i * dstStride] = srcData[i];
}

template<int N, class DTYPE> inline __device__ void
copy(DTYPE* dstData, const DTYPE* srcData) {
	#pragma unroll
	for(int i = 0; i < N; i++) 
		dstData[i] = srcData[i];
}
*/
//---- Extended Copy COMPLEX<->{float,float} ------------------------------
//- Can be used to separate components and reduce bank conflicts

// 2xfloat to complex

template<int N> inline __device__ void
copy(COMPLEX* dstData, int dstStride,
	const float* src_real, const float* src_imag, int srcStride = 1)
{
	#pragma unroll
	for(int i = 0; i < N; i++) {
		COMPLEX tmp;
		tmp.x = src_real[i * srcStride];
		tmp.y = src_imag[i * srcStride];
		dstData[i * dstStride] = tmp;
	}
}

template<int N> inline __device__ void
copy(COMPLEX* dstData, // int dstSride = 1,
	const float* src_real, const float* src_imag, int srcStride = 1) {
	copy<N>(dstData, 1, src_real, src_imag, srcStride);
}

// complex to 2xfloat

template<int N> inline __device__ void
copy(float* dst_real, float* dst_imag, int dstStride,
	const COMPLEX* srcData, int srcStride = 1)
{
	#pragma unroll
	for(int i = 0; i < N; i++) {
		COMPLEX tmp = srcData[i * srcStride];
		dst_real[i * dstStride] = tmp.x;
		dst_imag[i * dstStride] = tmp.y;
	}
}

template<int N> inline __device__ void
copy(float* dst_real, float* dst_imag, // dstStride = 1,
	const COMPLEX* srcData, int srcStride = 1) {
	copy<N>(dst_real, dst_imag, 1, srcData, srcStride);
}


//---- Extended Copy float.[x|y]<->float ------------------------------
//- Can be used to reduce shared memory using a single buffer

// 2xfloat to complex

template<int N> inline __device__ void
copyX(COMPLEX* dstData, int dstStride,
	const float* srcData, int srcStride = 1)
{
	#pragma unroll
	for(int i = 0; i < N; i++)
        dstData[i * dstStride].x = srcData[i * srcStride];
}

template<int N> inline __device__ void
copyX(COMPLEX* dstData, // int dstStride = 1,
	const float* srcData, int srcStride = 1) {
		copyX<N>(dstData, 1, srcData, srcStride);
}

template<int N> inline __device__ void
copyY(COMPLEX* dstData, int dstStride,
	const float* srcData, int srcStride = 1)
{
	#pragma unroll
	for(int i = 0; i < N; i++)
        dstData[i * dstStride].y = srcData[i * srcStride];
}

template<int N> inline __device__ void
copyY(COMPLEX* dstData, // int dstStride = 1,
	const float* srcData, int srcStride = 1) {
		copyY<N>(dstData, 1, srcData, srcStride);
}

// complex to 2xfloat

template<int N> inline __device__ void
copyX(float* dstData, int dstStride,
	const COMPLEX* srcData, int srcStride = 1)
{
	#pragma unroll
	for(int i = 0; i < N; i++)
		dstData[i * dstStride] = srcData[i * srcStride].x;
}

template<int N> inline __device__ void
copyX(float* dstData, // int dststride
	const COMPLEX* srcData, int srcStride = 1) {
		copyX<N>(dstData, 1, srcData, srcStride);
}

template<int N> inline __device__ void
copyY(float* dstData, int dstStride,
	const COMPLEX* srcData, int srcStride = 1)
{
	#pragma unroll
	for(int i = 0; i < N; i++)
		dstData[i * dstStride] = srcData[i * srcStride].y;
}

template<int N> inline __device__ void
copyY(float* dstData, // int dststride
	const COMPLEX* srcData, int srcStride = 1) {
		copyY<N>(dstData, 1, srcData, srcStride);
}

//- Templates para el intercambio de datos float: reg->shm->reg
//- Version para intercambio de datos dentro de *** warp ***
//- Sincroniza cuando hay multiplexion de la memoria compartida

// Version para array ShMem de tipo 'float'
template<int N, int PLX> inline __device__ void
shmExWarp(COMPLEX* regData, int locZ,
		float* srcTmp, const int srcStride,
		float* dstTmp, const int dstStride = 1)
{
	#pragma unroll
	for(int plex = 0; plex < PLX; plex++) {
		if(plex == locZ) {
			copyX<N>(srcTmp, srcStride, regData, 1);
			copyX<N>(regData, 1, dstTmp, dstStride);
			// Here __threadfence is not required
			copyY<N>(srcTmp, srcStride, regData, 1);
			copyY<N>(regData, 1, dstTmp, dstStride);
		}
		if(PLX > 1) __syncthreads(); else __threadfence_block(); /// Barrier
	}
}

// Version para array ShMem de tipo 'complex'
template<int N, int PLX> inline __device__ void
shmExWarp(COMPLEX* regData, int locZ,
		COMPLEX* srcTmp, const int srcStride,
		COMPLEX* dstTmp, const int dstStride = 1)
{
	#pragma unroll
	for(int plex = 0; plex < PLX; plex++) {
		if(plex == locZ) {
			copy<N>(srcTmp, srcStride, regData, 1);
			copy<N>(regData, 1, dstTmp, dstStride);
		}
		if(PLX > 1) __syncthreads(); else __threadfence_block(); /// Barrier
	}
}

//- Template para el intercambio de datos float: reg->shm->reg
//- Version para intercambio de datos dentro de *** bloque ***
//- Sincroniza cuando hay multiplexion de la memoria compartida

// Version para array ShMem de tipo 'float'
template<int N, int PLX> inline __device__ void
shmExBlk(COMPLEX* regData, int locZ,
		float* srcTmp, const int srcStride,
		float* dstTmp, const int dstStride = 1)
{
	#pragma unroll
	for(int plex = 0; plex < PLX; plex++) {
		if(plex == locZ) copyX<N>(srcTmp, srcStride, regData, 1);
		if(PLX > 1) __syncthreads(); else __threadfence_block(); /// Barrier

		if(plex == locZ) copyX<N>(regData, 1, dstTmp, dstStride);
		if(PLX > 1) __syncthreads(); else __threadfence_block(); /// Barrier

		if(plex == locZ) copyY<N>(srcTmp, srcStride, regData, 1);
		if(PLX > 1) __syncthreads(); else __threadfence_block(); /// Barrier

		if(plex == locZ) copyY<N>(regData, 1, dstTmp, dstStride);
		if(PLX > 1) __syncthreads(); else __threadfence_block(); /// Barrier
	}
}

// Version para array ShMem de tipo 'complex'
template<int N, int PLX> inline __device__ void
shmExBlk(COMPLEX* regData, int locZ,
		COMPLEX* srcTmp, const int srcStride,
		COMPLEX* dstTmp, const int dstStride = 1)
{
	#pragma unroll
	for(int plex = 0; plex < PLX; plex++) {
		if(plex == locZ) copy<N>(srcTmp, srcStride, regData, 1);
		if(PLX > 1) __syncthreads(); else __threadfence_block(); /// Barrier

		if(plex == locZ) copy<N>(regData, 1, dstTmp, dstStride);
		if(PLX > 1) __syncthreads(); else __threadfence_block(); /// Barrier
	}
}



//- Template para el intercambio de datos float: reg->shm->reg
//- Version para intercambio de datos dentro de *** warp ***
//- Sincroniza siempre al terminar cada sub-bloque

// Version para array ShMem de tipo 'float'
template<int N, int PLX> inline __device__ void
shmExSync(COMPLEX* regData, int locZ,
		float* srcTmp, const int srcStride,
		float* dstTmp, const int dstStride = 1)
{
	#pragma unroll
	for(int plex = 0; plex < PLX; plex++) {
		if(plex == locZ) copyX<N>(srcTmp, srcStride, regData, 1);
		__syncthreads();

		if(plex == locZ) copyX<N>(regData, 1, dstTmp, dstStride);
		__syncthreads();

		if(plex == locZ) copyY<N>(srcTmp, srcStride, regData, 1);
		__syncthreads();

		if(plex == locZ) copyY<N>(regData, 1, dstTmp, dstStride);
		__syncthreads();
	}
}

// Version para array ShMem de tipo 'complex'
template<int N, int PLX> inline __device__ void
shmExSync(COMPLEX* regData, int locZ,
		COMPLEX* srcTmp, const int srcStride,
		COMPLEX* dstTmp, const int dstStride = 1)
{
	#pragma unroll
	for(int plex = 0; plex < PLX; plex++) {
		if(plex == locZ) copy<N>(srcTmp, srcStride, regData, 1);
		__syncthreads();

		if(plex == locZ) copy<N>(regData, 1, dstTmp, dstStride);
		__syncthreads();
	}
}


//- Template para el intercambio de datos float: reg->shm->reg
//- Version para intercambio de datos dentro de *** bloque ***
//- Sincroniza siempre al terminar cada sub-bloque
//- Multiplexion deshabilitada (anterior con PLX = 1 y locZ = 0)

// Version para array ShMem de tipo 'float'
template<int N> inline __device__ void
shmExSync(COMPLEX* regData,
		float* srcTmp, const int srcStride,
		float* dstTmp, const int dstStride = 1)
{
	copyX<N>(srcTmp, srcStride, regData, 1);
	__syncthreads();

	copyX<N>(regData, 1, dstTmp, dstStride);
	__syncthreads();

	copyY<N>(srcTmp, srcStride, regData, 1);
	__syncthreads();

	copyY<N>(regData, 1, dstTmp, dstStride);
	__syncthreads();
}

// Version para array ShMem de tipo 'complex'
template<int N> inline __device__ void
shmExSync(COMPLEX* regData,
		COMPLEX* srcTmp, const int srcStride,
		COMPLEX* dstTmp, const int dstStride = 1)
{
	copy<N>(srcTmp, srcStride, regData, 1);
	__syncthreads();

	copy<N>(regData, 1, dstTmp, dstStride);
	__syncthreads();
}


// Float4 copy operatos (Tridiag Extension)
// ----------------------------------------------------------

template<int N> inline __device__ void
copyX(float4* dstData, int dstStride,
	const float* srcData, int srcStride = 1)
{
	#pragma unroll
	for(int i = 0; i < N; i++)
        dstData[i * dstStride].x = srcData[i * srcStride];
}

template<int N> inline __device__ void
copyY(float4* dstData, int dstStride,
	const float* srcData, int srcStride = 1)
{
	#pragma unroll
	for(int i = 0; i < N; i++)
        dstData[i * dstStride].y = srcData[i * srcStride];
}

template<int N> inline __device__ void
copyZ(float4* dstData, int dstStride,
	const float* srcData, int srcStride = 1)
{
	#pragma unroll
	for(int i = 0; i < N; i++)
        dstData[i * dstStride].z = srcData[i * srcStride];
}

template<int N> inline __device__ void
copyW(float4* dstData, int dstStride,
	const float* srcData, int srcStride = 1)
{
	#pragma unroll
	for(int i = 0; i < N; i++)
        dstData[i * dstStride].w = srcData[i * srcStride];
}

template<int N> inline __device__ void
copyX(float* dstData, int dstStride,
	const float4* srcData, int srcStride = 1)
{
	#pragma unroll
	for(int i = 0; i < N; i++)
        dstData[i * dstStride] = srcData[i * srcStride].x;
}

template<int N> inline __device__ void
copyY(float* dstData, int dstStride,
	const float4* srcData, int srcStride = 1)
{
	#pragma unroll
	for(int i = 0; i < N; i++)
        dstData[i * dstStride] = srcData[i * srcStride].y;
}

template<int N> inline __device__ void
copyZ(float* dstData, int dstStride,
	const float4* srcData, int srcStride = 1)
{
	#pragma unroll
	for(int i = 0; i < N; i++)
        dstData[i * dstStride] = srcData[i * srcStride].z;
}

template<int N> inline __device__ void
copyW(float* dstData, int dstStride,
	const float4* srcData, int srcStride = 1)
{
	#pragma unroll
	for(int i = 0; i < N; i++)
        dstData[i * dstStride] = srcData[i * srcStride].w;
}

template<int N> inline __device__ void
copyXY(float2* dstData, int dstStride,
	const float4* srcData, int srcStride = 1)
{
	#pragma unroll
	for(int i = 0; i < N; i++) {
		float4 srcTmp4 = srcData[i * srcStride];
		float2 dstTmp2 = make_float2(srcTmp4.x, srcTmp4.y);
        dstData[i * dstStride] = dstTmp2;
	}
}

template<int N> inline __device__ void
copyZW(float2* dstData, int dstStride,
	const float4* srcData, int srcStride = 1)
{
	#pragma unroll
	for(int i = 0; i < N; i++) {
		float4 srcTmp4 = srcData[i * srcStride];
		float2 dstTmp2 = make_float2(srcTmp4.z, srcTmp4.w);
        dstData[i * dstStride] = dstTmp2;
	}
}

template<int N> inline __device__ void
copyXY(float4* dstData, int dstStride,
	const float2* srcData, int srcStride = 1)
{
	#pragma unroll
	for(int i = 0; i < N; i++) {
		float2 srcTmp2 = srcData[i * srcStride];
        dstData[i * dstStride].x = srcTmp2.x;
		dstData[i * dstStride].y = srcTmp2.y;
	}
}

template<int N> inline __device__ void
copyZW(float4* dstData, int dstStride,
	const float2* srcData, int srcStride = 1)
{
	#pragma unroll
	for(int i = 0; i < N; i++) {
		float2 srcTmp2 = srcData[i * srcStride];
        dstData[i * dstStride].z = srcTmp2.x;
		dstData[i * dstStride].w = srcTmp2.y;
	}
}



/*
//- Operadores Copy con padding implicito

// Version para array ShMem de tipo 'float'
template<int N> inline __device__ void
shmCopy(COMPLEX* dstBase, const int dstDesp, const int dstStride,
		const COMPLEX* srcBase, const int srcDesp, const int srcStride)
{
	const int log2bnk = 5;
	#pragma unroll
	for(int i = 0; i < N; i++) {
		int srcPos = srcDesp + srcStride * i;
		int dstPos = dstDesp + dstStride * i;
		dstBase[dstPos + (dstPos >> log2bnk)] =
			srcBase[srcPos + (srcPos >> log2bnk)];
	}
}

// Version para array ShMem de tipo 'float'
template<int N> inline __device__ void
shmCopy(COMPLEX* dstBase, const int dstDesp, const int dstStride,
		const COMPLEX* srcReg)
{
	const int log2bnk = 5;
	#pragma unroll
	for(int i = 0; i < N; i++) {
		int dstPos = dstDesp + dstStride * i;
		dstBase[dstPos + (dstPos >> log2bnk)] = srcReg[i];
	}
}

// Version para array ShMem de tipo 'float'
template<int N> inline __device__ void
shmCopy(COMPLEX* dstReg,
		const COMPLEX* srcBase, const int srcDesp, const int srcStride)
{
	const int log2bnk = 5;
	#pragma unroll
	for(int i = 0; i < N; i++) {
		int srcPos = srcDesp + srcStride * i;
		dstReg[i] =	srcBase[srcPos + (srcPos >> log2bnk)];
	}
}

//- Operadores Exchange con padding implicito

// Version para array ShMem de tipo 'float'
template<int N> inline __device__ void
shmExchange(COMPLEX* regData, const int wrtDesp, const int wrtStride,
			COMPLEX* shmBase, const int readDesp, const int readStride)
{
	__syncthreads();
	shmCopy<N>(shmBase, wrtDesp, wrtStride, regData);
	__syncthreads();
	shmCopy<N>(regData, shmBase, readDesp, readStride);
}
*/

#endif // _OP_COPY

