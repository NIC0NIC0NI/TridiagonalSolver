//- =======================================================================
//+ KTridiag LF algorithm 
//- =======================================================================



#include "tools/cudacl.hxx"
#include "inc/op_copy.hxx"
#include "inc/op_twiddle.hxx"

#include "KLauncher3.hxx"
#include "inc/op_reduce.hxx"
#include "inc/op_mixr.hxx"
#include "inc/op_load.hxx"
#include "inc/op_shfl.hxx"

//---- Include Section ---------------------------------------------------
#include <cstdio>
//---- Interface Functions -----------------------------------------------




#ifndef __CUDA_ARCH__
    #define __CUDA_ARCH__ CUDA_ARCH
#endif


#if  __CUDA_ARCH__ < 400
	#define tabla3 triXf32A
#endif

#if  __CUDA_ARCH__ >= 400
	#define tabla3 triXf32B
#endif




//---- Butterfly operator ------------------------------------------------

template<class DTYPE, int size> struct butterfly {

	//- The first butterfly step is an optimized version of 'butterfly_step'
	static inline __device__ void
	init(Eqn<DTYPE>* E1, Eqn<DTYPE>* E2, int s = 1);

	//- Generic butterfly step, is more efficient to call 'butterfly_init' first
	static inline __device__ void
	step(Eqn<DTYPE>* E1, Eqn<DTYPE>* E2, int s = 1);

};

template<class DTYPE> struct butterfly<DTYPE, 2> {

	
	//- The initial Rad<2> case is optimized
	static inline __device__ void
	init(Eqn<DTYPE>* E1, Eqn<DTYPE>* E2, int s = 1) {

		E2[0]=E1[0];
		E2[1]=E1[1];


		Eqn<DTYPE> eqA = reduce(E1[0],E2[1],2);
		Eqn<DTYPE> eqB = reduce(E1[0],E2[1],0);

		E1[1] = reduce(eqB,E1[1],11);
		E2[1] = reduce(eqA,E2[0],7);		

		
	}
	



	//- The general Rad<2> case is defined according to the equation
	static inline __device__ void
	step(Eqn<DTYPE>* E1, Eqn<DTYPE>* E2, int s = 1) {
		

		Eqn<DTYPE> eqA = reduce(E1[0],E2[1],2);
		Eqn<DTYPE> eqB = reduce(E1[0],E2[1],0);

		E1[1] = reduce(eqB,E1[1],11);
		E2[1] = reduce(eqA,E2[0],7);
		
	}
};




//---- Radix operator ----------------------------------------------------

//- Generic radix stage, used in the main loop of the algorithm
template<int RAD, class DTYPE> inline __device__ void
radix(Eqn<DTYPE>* L, Eqn<DTYPE>* C) {
	butterfly<DTYPE, RAD>::step(L, C);
}

//- Mixed-radix stage, only called once before the main loop
template<int SIZE, int RAD, class DTYPE> inline __device__ void
radix(Eqn<DTYPE>* L, Eqn<DTYPE>* C) {
	#pragma unroll
	for(int i = 0; i < SIZE; i+= RAD)
		butterfly<DTYPE, RAD>::init(L+i, C+i);
}








template<class DTYPE, int SIZE> struct store {

	//- General version signature
	static inline __device__ void
	col(DTYPE* __restrict__ dst, const DTYPE* __restrict__ src);
	

};

template<class DTYPE> struct store<DTYPE, 2> {
	//- Specialization to load 2 consecutive elements using float2
	static inline __device__ void
	col(DTYPE* __restrict__ dst, const DTYPE* __restrict__ src)
	{
		//typename simd<DTYPE, 2>::type t = *(typename simd<DTYPE, 2>::type*)(src);
		*(typename simd<DTYPE, 2>::type*)(dst) = *(typename simd<DTYPE, 2>::type*)(src);
	}
};






template<class DTYPE, int N, int DIR, int RAD, int SHM> __global__ void
KTridiagLF(const DTYPE* __restrict__ srcL, 
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
	Eqn<DTYPE> form1[RAD], form2[RAD];

	//Adaptar a LF - SH
	__shared__ Eqn<DTYPE> shm[N > RAD ? SHM : 1];
	__shared__ Eqn<DTYPE> shm2[N > RAD ? SHM : 1];

	// Load 'regC'. Left and right equations are initialized by 'radix_init'

	load<DTYPE, RAD>::col(form1, 'x', srcL + glbRPos);
	load<DTYPE, RAD>::col(form1, 'y', srcC + glbRPos);
	load<DTYPE, RAD>::col(form1, 'z', srcR + glbRPos);
	load<DTYPE, RAD>::col(form1, 'w', dstX + glbRPos);
	
	// The first radix stage is always executed
	const int MIXRAD = MIXR<N, RAD>::val;	
	radix<RAD, MIXRAD>(form1,form2);


	const int warpSize=32;
	const int warp_id = threadIdx.x / warpSize;
	const int widthSize= (N>64) ? 32 : (N/RAD);



	int i=1;
	#pragma unroll
	for(int width=2;width <= widthSize;width*=2) 
	{
		int lane_id = (threadId&(width-1));
		Eqn<DTYPE> Eq1 = shfl<DTYPE>::shfl_Eq(form1[1],i-1,width);
		Eqn<DTYPE> Eq2 = shfl<DTYPE>::shfl_Eq(form2[1],i-1,width);
		
		
		if (lane_id > (i-1)){

			
			
			Eqn<DTYPE> EqAux1 = reduce(Eq1,(form2[0]),2); 
			Eqn<DTYPE> EqAux2 = reduce(Eq1,form2[0],0);
			
			form1[0] = reduce(EqAux2,form1[0],11);
			form2[0] = reduce(EqAux1,Eq2,7);			
			
			Eqn<DTYPE> EqAuxA = reduce(Eq1,form2[1],2); 
			Eqn<DTYPE> EqAuxB = reduce(Eq1,form2[1],0);

			form1[1] = reduce(EqAuxB,form1[1],11);
			form2[1] = reduce(EqAuxA,Eq2,7); 	
		
		}
		i*=2;

	}

	
	
	// Process the remaining stages
	#pragma unroll
	for(int cont = widthSize, accRad = RAD*widthSize; accRad < N; cont = accRad, accRad *= RAD) {

		const int maskLo1 = cont - 1, maskHi1 = ~maskLo1;
		const int writeIndex = RAD*(threadId&maskHi1)+maskLo1+(threadId&maskLo1)+1;
		int str1 = 0;
		
		
		const int maskLo2= accRad-1, maskHi2= ~maskLo2;
		const int readIndex= RAD*(threadId&maskHi2)+maskLo2;
		int str2 = (threadId&maskLo2)+1;
		
		int offset = 0; 
	
		if(cont==widthSize)
		{		
			copy<2>(shm+shmOffset+2*threadId,1,form1);
			copy<2>(shm2+shmOffset+2*threadId,1,form2);
	
		}else{		
			copy<RAD/2>(shm+shmOffset+writeIndex,str1,&form1[1],2);
			copy<RAD/2>(shm2+shmOffset+writeIndex,str1,&form2[1],2);
		}
		
		__syncthreads();

		copy<RAD,2>(form1,shm+shmOffset+readIndex,str2, offset); 		
		copy<RAD,2>(form2,shm2+shmOffset+readIndex,str2,offset);

		radix<RAD>(form1,form2);

	}


	if(threadIdx.x==(blockDim.x-1))
	{			
		dstX[verticalId * stride]=form2[RAD-1].w/form2[RAD-1].y; 
	}	

	__syncthreads();

	const DTYPE x0= dstX[verticalId * stride];
	if((i*glbWStr+1)<N)	
		dstX[glbWPos+(RAD-1)*glbWStr+1]=(form2[RAD-1].w - (x0 ) * form2[RAD-1].y) / form2[RAD-1].z; 

	copy<RAD-1>(form2,shm2+shmOffset+threadId,blockDim.x);
	
	#pragma unroll
	for(int i=0;i<(RAD-1);i++)
	{	
			
			dstX[glbWPos + i * glbWStr+1]= (form2[i].w - x0 * form2[i].y) / form2[i].z;
		
	}	

}



 // Not shuffle implementation
 
template<class DTYPE, int N, int DIR, int RAD, int SHM> __global__ void
KTridiagLF2(const DTYPE* __restrict__ srcL, 
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
	Eqn<DTYPE> form1[RAD], form2[RAD];


	// Load 'regC'. Left and right equations are initialized by 'radix_init'
	
	load<DTYPE, RAD>::col(form1, 'x', srcL + glbRPos);
	load<DTYPE, RAD>::col(form1, 'y', srcC + glbRPos);
	load<DTYPE, RAD>::col(form1, 'z', srcR + glbRPos);
	load<DTYPE, RAD>::col(form1, 'w', dstX + glbRPos);
	
	// The first radix stage is always executed
	const int MIXRAD = MIXR<N, RAD>::val;


	
	
	radix<RAD, MIXRAD>(form1,form2);





	const int warpSize=32;
	const int warp_id = threadIdx.x / warpSize;
	

	const int widthSize= (N/RAD);


	int i=1;
	#pragma unroll
	for(int width=2;width <= widthSize;width*=2) 
	{
		int lane_id = (threadId&(width-1));
		Eqn<DTYPE> Eq1 = shfl<DTYPE>::shfl_Eq(form1[1],i-1,width);
		Eqn<DTYPE> Eq2 = shfl<DTYPE>::shfl_Eq(form2[1],i-1,width);
		
		
		if (lane_id > (i-1)){
		
			Eqn<DTYPE> EqAux1 = reduce(Eq1,(form2[0]),2); 
			Eqn<DTYPE> EqAux2 = reduce(Eq1,form2[0],0);
			
			form1[0] = reduce(EqAux2,form1[0],11);
			form2[0] = reduce(EqAux1,Eq2,7);			
			
			Eqn<DTYPE> EqAuxA = reduce(Eq1,form2[1],2); 
			Eqn<DTYPE> EqAuxB = reduce(Eq1,form2[1],0);

			form1[1] = reduce(EqAuxB,form1[1],11);
			form2[1] = reduce(EqAuxA,Eq2,7); 	
		
		}
		i*=2;
		
	}

		
	const DTYPE x = form2[1].w/form2[1].y; 

	if(threadId==(widthSize-1))	
		dstX[verticalId * stride]=x; 

	
	DTYPE x0 = shfl<DTYPE>::shflDTYPE(x,widthSize-1,32);

	
	DTYPE X[RAD];
	X[0] = (form2[0].w - x0 * form2[0].y) / form2[0].z;
	X[1] = (form2[1].w - x0 * form2[1].y) / form2[1].z;


	
	if(threadId!=(widthSize-1))
		store<DTYPE, RAD>::col( dstX + glbWPos + threadId /*+ 1*/ , X);
	else
		dstX[glbWPos+threadId]=X[0];	
	
}




// --- BranchTable -------------------------------------------------------


//- Template instantiation and branchtable for 'float' kernels in Kepler Architecture
const static kernelCfg<float> triXf32A[] = { //! GPU dependent
	NULL_ROW(1),
	NULL_ROW(2),
	ROW(KTridiagLF2, float,    4, 128, 2),
	ROW(KTridiagLF2, float,    8, 128, 2),//
	ROW(KTridiagLF2, float,   16, 128, 2),//
	ROW(KTridiagLF2, float,   32, 128, 2),//
	ROW(KTridiagLF2, float,   64, 128, 2),//
	ROW(KTridiagLF, float,  128, 128, 2),//
	ROW(KTridiagLF, float,  256, 256, 2),//
	ROW(KTridiagLF, float,  512, 512, 2),//
	ROW(KTridiagLF, float, 1024,1024, 2),//
	NULL_ROW(4096),
};

//Maxwell configuration
const static kernelCfg<float> triXf32B[] = { //! GPU dependent
	NULL_ROW(1),
	NULL_ROW(2),
	ROW(KTridiagLF2, float,    4, 128, 2),
	ROW(KTridiagLF2, float,    8, 128, 2),//
	ROW(KTridiagLF2, float,   16, 128, 2),//
	ROW(KTridiagLF2, float,   32, 128, 2),//
	ROW(KTridiagLF2, float,   64, 128, 2),//
	ROW(KTridiagLF, float,  128, 256, 2),
	ROW(KTridiagLF, float,  256, 512, 2),//
	ROW(KTridiagLF, float,  512, 512, 2),//
	ROW(KTridiagLF, float, 1024,1024, 2),//
	NULL_ROW(4096),
};






//- Main library function for 'float' equations
int KTridiagLF(float* data, int dir, int N, int M, int batch) {

	
	if (N > 1024)
		return -1;

	return KLauncher3(tabla3, sizeof(tabla3), data, dir, N, batch);
}


