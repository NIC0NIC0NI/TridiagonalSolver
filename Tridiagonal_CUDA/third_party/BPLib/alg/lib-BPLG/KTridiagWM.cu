//- =======================================================================
//+ KTridiag Wang&Mou algorithm 
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "tools/cudacl.hxx"
#include "inc/op_copy.hxx"
#include "inc/op_twiddle.hxx"

#include "inc/op_mixr.hxx"
#include "inc/op_load.hxx"
#include "inc/op_reduce.hxx"

#include "KLauncher3.hxx"

#ifndef __CUDA_ARCH__
    #define __CUDA_ARCH__ CUDA_ARCH
#endif


#if  __CUDA_ARCH__ < 400
	#define tabla2 triXf32A
#endif

#if  __CUDA_ARCH__ >= 400
	#define tabla2 triXf32B
#endif


//---- Butterfly operator ------------------------------------------------

template<class DTYPE, int size> struct butterfly {

	//- The first butterfly step is an optimized version of 'butterfly_step'
	static inline __device__ void
	init(Eqn<DTYPE>* L, Eqn<DTYPE>* C, Eqn<DTYPE>* R, int s = 1);

	//- Generic butterfly step, is more efficient to call 'butterfly_init' first
	static inline __device__ void
	step(Eqn<DTYPE>* L, Eqn<DTYPE>* C, Eqn<DTYPE>* R, int s = 1);

};

template<class DTYPE> struct butterfly<DTYPE, 2> {

	//- The initial Rad<2> case is optimized
	static inline __device__ void
	init(Eqn<DTYPE>* L, Eqn<DTYPE>* C, Eqn<DTYPE>* R, int s = 1) {
		Eqn<DTYPE> eqL = reduce(C[0], C[s], 0);
		Eqn<DTYPE> eqR = reduce(C[s], C[0],10);

		L[0] = eqL; L[s] = eqL;
		C[0] = eqL; C[s] = eqR;
		R[0] = eqR; R[s] = eqR;
	}

	//- The general Rad<2> case is defined according to the equation
	static inline __device__ void
	step(Eqn<DTYPE>* L, Eqn<DTYPE>* C, Eqn<DTYPE>* R, int s = 1) {
		Eqn<DTYPE> eq3 = reduce(R[0], L[s], 2);
		Eqn<DTYPE> eqL = reduce(L[0], eq3 , 1); /// OPT

		Eqn<DTYPE> eq1 = reduce(L[s], R[0], 8);
		Eqn<DTYPE> eqR = reduce(R[s], eq1 , 9); /// OPT

		Eqn<DTYPE> eqA = reduce(C[0], eq3 , 1);
		Eqn<DTYPE> eqB = reduce(C[s], eq1 , 9);

		L[0] = eqL; L[s] = eqL;
		C[0] = eqA; C[s] = eqB;
		R[0] = eqR; R[s] = eqR;
	}
};


template<class DTYPE> struct butterfly<DTYPE, 4> {

	//- The initial Rad<4> case is optimized
	static inline __device__ void
	init(Eqn<DTYPE>* L, Eqn<DTYPE>* C, Eqn<DTYPE>* R, int s = 1) {
		const int s0 = 0, s1 = s, s2 = 2 * s, s3 = 3 * s;
		Eqn<DTYPE> eq1 = reduce(C[s1],C[s0], 10);
		Eqn<DTYPE> eq2 = reduce(C[s2],eq1  , 10);
		Eqn<DTYPE> eqR = reduce(C[s3],eq2  , 10);
		Eqn<DTYPE> eqB = reduce(C[s3],eq2  ,  8);

		Eqn<DTYPE> eq3 = reduce(C[s2], C[s3], 0);
		Eqn<DTYPE> eq4 = reduce(C[s1], eq3  , 0);
		Eqn<DTYPE> eqL = reduce(C[s0], eq4  , 0);
		Eqn<DTYPE> eqA = reduce(C[s0], eq4  , 2);

		L[s0] = eqL; L[s1] = eqL; L[s2] = eqL; L[s3] = eqL;
		C[s0] = eqL; C[s1] = eqA; C[s2] = eqB; C[s3] = eqR; 
		R[s0] = eqR; R[s1] = eqR; R[s2] = eqR; R[s3] = eqR;
	}

	//- The general Rad<4> case is defined according to the equation
	static inline __device__ void
	step(Eqn<DTYPE>* L, Eqn<DTYPE>* C, Eqn<DTYPE>* R, int s = 1) {
		const int s0 = 0, s1 = s, s2 = 2 * s, s3 = 3 * s;

		Eqn<DTYPE> eq1 = reduce(L[s1], R[s0], 8); 
		Eqn<DTYPE> eq2 = reduce(R[s1], eq1  , 9); 
		Eqn<DTYPE> eq3 = reduce(L[s2], eq2  , 8);
		Eqn<DTYPE> eq4 = reduce(R[s2], eq3  , 9);
		Eqn<DTYPE> eq5 = reduce(L[s3], eq4  , 8);
		Eqn<DTYPE> eqR = reduce(R[s3], eq5  , 9); /// OPT
		Eqn<DTYPE> eqD = reduce(C[s3], eq5  , 9);
	
		Eqn<DTYPE> eq7 = reduce(R[s2], L[s3], 2);
		Eqn<DTYPE> eq8 = reduce(L[s2], eq7  , 1);
		Eqn<DTYPE> eq9 = reduce(R[s1], eq8  , 2);
		Eqn<DTYPE> eq10= reduce(L[s1], eq9  , 1);
		Eqn<DTYPE> eq11= reduce(R[s0], eq10 , 2);
		Eqn<DTYPE> eqL = reduce(L[s0], eq11 , 1); /// OPT
		Eqn<DTYPE> eqA = reduce(C[s0], eq11 , 1);

		Eqn<DTYPE> eqU = reduce(C[s1], eq1  ,  9);
		Eqn<DTYPE> eqV = reduce(eq9  , eq1  , 10);
		Eqn<DTYPE> eqB = reduce(eqU  , eqV  ,  1);
	
		Eqn<DTYPE> eqM = reduce(C[s2], eq3  ,  9);
		Eqn<DTYPE> eqN = reduce(eq7  , eq3  , 10);
		Eqn<DTYPE> eqC = reduce(eqM  , eqN  ,  1);

		L[s0] = eqL; L[s1] = eqL; L[s2] = eqL; L[s3] = eqL;
		C[s0] = eqA; C[s1] = eqB; C[s2] = eqC; C[s3] = eqD;
		R[s0] = eqR; R[s1] = eqR; R[s2] = eqR; R[s3] = eqR;
	}
};

template<class DTYPE> struct butterfly<DTYPE, 8> {

	//- The initial Rad<8> case is recursively defined
	static inline __device__ void
	init(Eqn<DTYPE>* L, Eqn<DTYPE>* C, Eqn<DTYPE>* R, int s = 1) {
		butterfly<DTYPE, 4>::init(L  , C  , R  , 1);
		butterfly<DTYPE, 4>::init(L+4, C+4, R+4, 1);

		butterfly<DTYPE, 2>::step(L  , C  , R  , 4);
		butterfly<DTYPE, 2>::step(L+1, C+1, R+1, 4);
		butterfly<DTYPE, 2>::step(L+2, C+2, R+2, 4);
		butterfly<DTYPE, 2>::step(L+3, C+3, R+3, 4);
	}

	//- The general Rad<8> case is recursively defined
	static inline __device__ void
	step(Eqn<DTYPE>* L, Eqn<DTYPE>* C, Eqn<DTYPE>* R, int s = 1) {
		butterfly<DTYPE, 4>::step(L  , C  , R  , 1);
		butterfly<DTYPE, 4>::step(L+4, C+4, R+4, 1);

		butterfly<DTYPE, 2>::step(L  , C  , R  , 4);
		butterfly<DTYPE, 2>::step(L+1, C+1, R+1, 4);
		butterfly<DTYPE, 2>::step(L+2, C+2, R+2, 4);
		butterfly<DTYPE, 2>::step(L+3, C+3, R+3, 4);
	}
};


//---- Radix operator ----------------------------------------------------

//- Generic radix stage, used in the main loop of the algorithm
template<int RAD, class DTYPE> inline __device__ void
radix(Eqn<DTYPE>* L, Eqn<DTYPE>* C, Eqn<DTYPE>* R) {
	butterfly<DTYPE, RAD>::step(L, C, R);
}

//- Mixed-radix stage, only called once before the main loop
template<int SIZE, int RAD, class DTYPE> inline __device__ void
radix(Eqn<DTYPE>* L, Eqn<DTYPE>* C, Eqn<DTYPE>* R) {
	#pragma unroll
	for(int i = 0; i < SIZE; i+= RAD)
		butterfly<DTYPE, RAD>::init(L+i, C+i, R+i);
}



//------- Cuda Kernels ---------------------------------------------------

//- Kernel for tridiagonal equation systems that fit in Shared Memory
//? Todo: Allow configurable 'stride' for other distributions
template<class DTYPE, int N, int DIR, int RAD, int SHM> __global__ void
KTridiagWM(const DTYPE* __restrict__ srcL, 
		 const DTYPE* __restrict__ srcC,
		 const DTYPE* __restrict__ srcR,
		       DTYPE* dstX, int stride)
{
	// Obtain group-1D, thread-X and batch-Y identifiers
	int groupId = get_group_id(0) + get_group_id(1) * get_num_groups(0);
	int threadId = get_local_id(0);      // Thread horizontal (N)
	int batchId = get_local_id(1);       // Thread vertical (batch)

	int verticalId = groupId * get_local_size(1) + batchId;

	// Offset for accesing thread data
	int shmOffset = batchId * N;

	int glbRPos = verticalId * stride + threadId * RAD;
	int glbWPos = verticalId * stride + threadId;
	int glbWStr = N / RAD;

	// Statically allocate registers and shared memory
	Eqn<DTYPE> regL[RAD], regC[RAD], regR[RAD];
	//? Test: What happens when allocating a ShMem array per coefficient
	__shared__ Eqn<DTYPE> shm[N > RAD ? SHM : 1];

	// Load 'regC'. Left and right equations are initialized by 'radix_init'
	//? Test: ShMem exchange instead of consecutive load for larger types
	load<DTYPE, RAD>::col(regC, 'x', srcL + glbRPos);
	load<DTYPE, RAD>::col(regC, 'y', srcC + glbRPos);
	load<DTYPE, RAD>::col(regC, 'z', srcR + glbRPos);
	load<DTYPE, RAD>::col(regC, 'w', dstX + glbRPos);
	
	// The first radix stage is always executed
	const int MIXRAD = MIXR<N, RAD>::val;
	radix<RAD, MIXRAD>(regL, regC, regR);
	
	// Process the remaining stages
	#pragma unroll
	for(int cont = 1, accRad = MIXRAD; accRad < N; cont = accRad, accRad *= RAD) {
		
		// Compute offset and stride for ShMem write
		int maskLo1 = cont - 1, maskHi1 = ~maskLo1;
		int off1 = shmOffset + (threadId & maskLo1) + RAD * (threadId & maskHi1);
		int str1 = cont;	// 1, R, R^2...

		// Compute offset and stride for ShMem read
		int maskLo2 = accRad - 1, maskHi2 = ~maskLo2;
		int off2 = shmOffset + (threadId & maskLo2) + RAD * (threadId & maskHi2);
		int str2 = accRad;	// R, R^2, R^3...
		
		// Masks for single write and multiple reads (triad optimization)
		int posLo = shmOffset + RAD * (threadId & maskHi2);
		int posHi = posLo + maskLo2;
		
		// Perform shared memory data exchange
		if(cont > 1) __syncthreads();
		copy<RAD>(shm + off1, str1, regC);
		__syncthreads();
		copy<RAD>(regC, shm + off2, str2);
		copy<RAD>(regL, shm + posLo, str2);
		copy<RAD>(regR, shm + posHi, str2);

		// Computation stage
		radix<RAD>(regL, regC, regR);
	}

	// Compute and store the final result in global memory to 'dstX'
	#pragma unroll
	for(int i = 0; i < RAD; i++) {
		const int d = glbWPos + glbWStr * i;
		dstX[d] = regC[i].w / regC[i].y;
	}

}

// --- BranchTable -------------------------------------------------------






const static kernelCfg<float> triXf32A[] = { //! GPU dependent
	NULL_ROW(1),
	NULL_ROW(2),
	ROW(KTridiagWM, float,    4, 256, 2),
	ROW(KTridiagWM, float,    8, 256, 2),
	ROW(KTridiagWM, float,   16, 256, 4),
	ROW(KTridiagWM, float,   32, 256, 2),
	ROW(KTridiagWM, float,   64, 256, 4),
	ROW(KTridiagWM, float,  128, 256, 4),
	ROW(KTridiagWM, float,  256, 256, 4),
	ROW(KTridiagWM, float,  512, 512, 4), //? Better using Rad8?
	ROW(KTridiagWM, float, 1024,1024, 4),
	ROW(KTridiagWM, float, 2048,2048, 4),
	NULL_ROW(4096),
};


const static kernelCfg<float> triXf32B[] = { //! GPU dependent
	NULL_ROW(1),
	NULL_ROW(2),
	ROW(KTridiagWM, float,    4, 256, 2),
	ROW(KTridiagWM, float,    8, 256, 2),
	ROW(KTridiagWM, float,   16, 256, 4),
	ROW(KTridiagWM, float,   32, 256, 2),
	ROW(KTridiagWM, float,   64, 128, 4),
	ROW(KTridiagWM, float,  128, 128, 4),
	ROW(KTridiagWM, float,  256, 512, 4),
	ROW(KTridiagWM, float,  512, 512, 4), //? Better using Rad8?
	ROW(KTridiagWM, float, 1024,1024, 4),
	ROW(KTridiagWM, float, 2048,2048, 4),
	NULL_ROW(4096),
};





//---- Interface Functions -----------------------------------------------

//- Main library function for 'float' equations
int KTridiagWM(float* data, int dir, int N, int M, int batch) {

	if(N>2048)
		return -1;

	return KLauncher3(tabla2, sizeof(tabla2), data, dir, N, batch);
}

