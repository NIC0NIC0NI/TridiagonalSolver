//- =======================================================================
//+ KTridiag CR algorithm (standalone version)
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "tools/cudacl.hxx"
#include "inc/op_copy.hxx"
#include "inc/op_twiddle.hxx"

#include "inc/op_mixr.hxx"
#include "inc/op_reduce.hxx"
#include "inc/op_load.hxx"
#include "inc/op_shfl.hxx"

#include "KLauncher3.hxx"

#ifndef __CUDA_ARCH__
    #define __CUDA_ARCH__ CUDA_ARCH
#endif


#if  __CUDA_ARCH__ < 400
	#define tabla triXf32A
#endif

#if  __CUDA_ARCH__ >= 400
	#define tabla triXf32B
#endif


//---- Butterfly operator ------------------------------------------------

template<class DTYPE, int size> struct butterfly {

	//- The first butterfly step is an optimized version of 'butterfly_step'
	static inline __device__ void
	init(Eqn<DTYPE>* E1, int s = 1);

	//- Generic butterfly step, is more efficient to call 'butterfly_init' first
	static inline __device__ void
	step(Eqn<DTYPE>* E1, int s = 1);

};

template<class DTYPE> struct butterfly<DTYPE, 2> {

	
	//- The initial Rad<2> case is optimized
	static inline __device__ void
	init(Eqn<DTYPE>* E1, int s = 1) {

		
	}
	


	//- The general Rad<2> case is defined according to the equation
	static inline __device__ void
	step(Eqn<DTYPE>* E1, int s = 1) {
		

		 Eqn<DTYPE> eq1 = reduce(E1[0],E1[1],2);
		 Eqn<DTYPE> eqR = reduce(eq1,E1[2], 0);

		 E1[1]=eqR;



	}
};


//---- Radix operator ----------------------------------------------------

//- Generic radix stage, used in the main loop of the algorithm
template<int RAD, class DTYPE> inline __device__ void
radix(Eqn<DTYPE>* C) {
	butterfly<DTYPE, RAD>::step( C);
}

//- Mixed-radix stage, only called once before the main loop
template<int SIZE, int RAD, class DTYPE> inline __device__ void
radix(Eqn<DTYPE>* C) {
	#pragma unroll
	for(int i = 0; i < SIZE; i+= RAD)
		butterfly<DTYPE, RAD>::init(C+i);
}


__device__ int isMultipleOf(const int N, const int value)
{


	int bit = ~0;
	int mask = bit<<(N);
	int mask2 = ~mask;
	if(!(value&mask2))
		return 1;

	return 0;
}













//------- Cuda Kernels ---------------------------------------------------

//- Kernel for tridiagonal equation systems that fit in Shared Memory
//? Todo: Allow configurable 'stride' for other distributions





// X in global memory, N larger than 64
template<class DTYPE, int N, int DIR, int RAD, int SHM> __global__ void
KTridiagCR(const DTYPE* __restrict__ srcL, 
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
	int glbWPos = verticalId * stride /*+ threadId*/;
	int glbWStr = N / RAD;

	// Statically allocate registers and shared memory
	Eqn<DTYPE> regC[3];
	//? Test: What happens when allocating a ShMem array per coefficient
	__shared__ Eqn<DTYPE> shm[N > RAD ? SHM : 1];

	// Load 'regC'. Left and right equations are initialized by 'radix_init'
	//? Test: ShMem exchange instead of consecutive load for larger types
	load<DTYPE, RAD>::col(regC, 'x', srcL + glbRPos);
	load<DTYPE, RAD>::col(regC, 'y', srcC + glbRPos);
	load<DTYPE, RAD>::col(regC, 'z', srcR + glbRPos);
	load<DTYPE, RAD>::col(regC, 'w', dstX + glbRPos);
	
	copy<RAD>(shm+RAD*threadId+shmOffset,1,regC);
	__syncthreads();

	if(threadIdx.x!=(blockDim.x-1))
		copy<1>(&regC[2],&shm[shmOffset+RAD*threadId+2],0);//No last thread
	else regC[2]=regC[1];

	radix<RAD>(regC);	

	// The first radix stage is always executed
	const int MIXRAD = MIXR<N, RAD>::val;
	int num_threads= blockDim.x;
	
	
	const int warpSize=32;
	const int warpAttack = warpSize;


	
	
	#pragma unroll
	for(int accRad = MIXRAD; accRad < (N/warpAttack);  accRad *= RAD) {

		int cont = accRad;
		int strideW = cont;
		int indexW = strideW*threadId + strideW -1+shmOffset;
		int strideR = 2*cont;
		int indexR = strideR * threadId + strideR-1+shmOffset;
		
		

		if(cont > 1) __syncthreads();
	
		if(threadId<num_threads)
			copy<1>(shm+indexW, 0, &regC[1]);

		__syncthreads();		
		num_threads/=2;
		
		if(threadId<num_threads)						
			copy<3>(regC,shm+indexR-cont,cont,((indexR-shmOffset+cont)>=N));
		radix<RAD>(regC);
	}



	if (threadId<warpSize)
	{
	
		
		//shuffling reduction
		int i,j;
		for(i=1, j=1; i < (warpSize/2) ; i*=2,j++)		
		{

			Eqn<DTYPE> EqUp = shfl<DTYPE>::shfl_Eq_Up(regC[1],i,warpSize);
			Eqn<DTYPE> EqDown =shfl<DTYPE>::shfl_Eq_Down(regC[1],i,warpSize);

			if(isMultipleOf(j,threadId+1))
			{
				Eqn<DTYPE> eq1 = reduce(EqUp,regC[1],2);
		 		Eqn<DTYPE> eqR = reduce(eq1,EqDown, 0);

		 		regC[1]=eqR;
																		
			}
		
		}

		
		
		//exchanging

		Eqn<DTYPE> EqUp = shfl<DTYPE>::shfl_Eq_Up(regC[1],i,warpSize);
		Eqn<DTYPE> EqDown = shfl<DTYPE>::shfl_Eq_Down(regC[1],i,warpSize);

		DTYPE x = 0;
		
		
		if(threadId==(warpSize/2 -1))
			{	
				Eqn<DTYPE> eqA, eqB;
				eqA=regC[1];
				eqB=EqDown;
				DTYPE tmp = (eqB.y*eqA.y)-(eqA.z*eqB.x);		
				x = (eqB.y*eqA.w - eqA.z*eqB.w)/tmp;	
			}
		if(threadId==(warpSize-1))
			{
				Eqn<DTYPE> eqA, eqB;	
				eqB = regC[1];
				eqA = EqUp;

				DTYPE tmp = (eqB.y*eqA.y)-(eqA.z*eqB.x);
				x = (eqB.w*eqA.y - eqA.w*eqB.x)/tmp;	
				
			}

			


		
	
		
		//shuffling substitution
		for(i=warpSize/4;i>0;i/=2)
		{
			j--;

			DTYPE xUp = shfl<DTYPE>::shfl_Up(x,i,warpSize);
			DTYPE xDown =shfl<DTYPE>::shfl_Down(x,i,warpSize);

			
			if(isMultipleOf(j,threadId-(i-1)))
			{
				
				Eqn<DTYPE> eq = regC[1];
				x=  (eq.w - eq.x*xUp - eq.z*xDown)/eq.y;
			}
	
		}
				
		dstX[glbWPos+((N/warpAttack))*(threadId+1)-1]=x;		
		
}	
	__syncthreads();
	num_threads=warpSize;
	
	
	#pragma unroll	
	for (int j = N/warpAttack; j > 1; j/=2)
   	{	

	       int delta = j/2;

	       __syncthreads();
	       if (threadId < num_threads)
	       {
		   int d = glbWPos;
		   int i = j * threadId + j/2 - 1;
		   Eqn<DTYPE> eq;
		   copy<1>(&eq,shm+shmOffset+i,1);
		 
		   if(i == delta - 1)
		         dstX[d+i] = (eq.w - eq.z*dstX[d+i+delta])/eq.y;
		   else
		         dstX[d+i] = (eq.w - eq.x*dstX[d+i-delta] - eq.z*dstX[d+i+delta])/eq.y;

			
		  		
		}
		num_threads *= 2;
     	}

	
}




// X in global memory, N smaller than 64

template<class DTYPE, int N, int DIR, int RAD, int SHM> __global__ void
KTridiagCR2(const DTYPE* __restrict__ srcL, 
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
	int glbWPos = verticalId * stride;
	int glbWStr = N / RAD;





	// Statically allocate registers and shared memory
	Eqn<DTYPE> regC[3];
	if(threadId!=(blockDim.x-1)){	
		load<DTYPE, 3>::col(regC, 'x', srcL + glbRPos);
		load<DTYPE, 3>::col(regC, 'y', srcC + glbRPos);
		load<DTYPE, 3>::col(regC, 'z', srcR + glbRPos);
		load<DTYPE, 3>::col(regC, 'w', dstX + glbRPos);
	}
	else
	{
		load<DTYPE, 2>::col(regC, 'x', srcL + glbRPos);
		load<DTYPE, 2>::col(regC, 'y', srcC + glbRPos);
		load<DTYPE, 2>::col(regC, 'z', srcR + glbRPos);
		load<DTYPE, 2>::col(regC, 'w', dstX + glbRPos);
		regC[2]=regC[1];
	}

	
	Eqn<DTYPE> eqUpInit = regC[0];

	radix<RAD>(regC);	


	// The first radix stage is always executed
	const int MIXRAD = MIXR<N, RAD>::val;
	int num_threads= blockDim.x;
	
	
	const int warpSize=(N>32)?32:(N/RAD);
	const int warpAttack = warpSize;


	
			
		//shuffling reduction
	int i,j;
	for(i=1, j=1; i < (warpSize/2) ; i*=2,j++)		
	{

		// i is increasing by x2
		// j is the logarithm of 2^x number
		
		Eqn<DTYPE> EqUp = shfl<DTYPE>::shfl_Eq_Up(regC[1],i,warpSize);
		Eqn<DTYPE> EqDown =shfl<DTYPE>::shfl_Eq_Down(regC[1],i,warpSize);

		//if thread+1 is multiple of 2^j then it modifies the equation
		
		
		if(isMultipleOf(j,threadId+1))
		{
			//Reduce ...

			
			
			Eqn<DTYPE> eq1 = reduce(EqUp,regC[1],2);
	 		Eqn<DTYPE> eqR = reduce(eq1,EqDown, 0);

	 		regC[1]=eqR;

																	
		}
		
		
	}

	
		
		//exchanging

		Eqn<DTYPE> EqUp = shfl<DTYPE>::shfl_Eq_Up(regC[1],i,warpSize);
		Eqn<DTYPE> EqDown = shfl<DTYPE>::shfl_Eq_Down(regC[1],i,warpSize);

		DTYPE x = 0;
		
		
			


			if(threadId==(warpSize/2 -1))
			{	
				Eqn<DTYPE> eqA, eqB;
				eqA=regC[1];
				eqB=EqDown;
				DTYPE tmp = (eqB.y*eqA.y)-(eqA.z*eqB.x);		
				x = (eqB.y*eqA.w - eqA.z*eqB.w)/tmp;	
				

			}
			if(threadId==(warpSize-1))
			{
				Eqn<DTYPE> eqA, eqB;	
				eqB = regC[1];
				eqA = EqUp;

				DTYPE tmp = (eqB.y*eqA.y)-(eqA.z*eqB.x);
				x = (eqB.w*eqA.y - eqA.w*eqB.x)/tmp;
				
			}
			
	
		
		//shuffling substitution
		for(i=warpSize/4;i>0;i/=2)
		{
			j--;

			DTYPE xUp = shfl<DTYPE>::shfl_Up(x,i,warpSize);
			DTYPE xDown =shfl<DTYPE>::shfl_Down(x,i,warpSize);

			
			if(isMultipleOf(j,threadId-(i-1)))
			{
				//substitution
				Eqn<DTYPE> eq = regC[1];
				x= (eq.w - eq.x*xUp - eq.z*xDown)/eq.y;
			}

			
	
		}

		
		Eqn<DTYPE> eq = eqUpInit;
		DTYPE xUp = shfl<DTYPE>::shfl_Up(x,1,warpSize);
		DTYPE x2 = (eq.w -eq.x*xUp - eq.z*x)/eq.y;

		dstX[glbWPos +((N/warpAttack))*(threadId+1)-2]=x2;		

		dstX[glbWPos+((N/warpAttack))*(threadId+1)-1]=x;
		
		

	
}



// X is in SHM, N smaller than 64 

template<class DTYPE, int N, int DIR, int RAD, int SHM> __global__ void
KTridiagCR3(const DTYPE* __restrict__ srcL, 
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
	int glbWPos = verticalId * stride /*+ threadId*/;
	int glbWStr = N / RAD;

	// Statically allocate registers and shared memory
	Eqn<DTYPE> regC[3];
	__shared__ DTYPE X[N> RAD ? SHM : 1];

	// Load 'regC'. Left and right equations are initialized by 'radix_init'
	
	if(threadId!=(blockDim.x-1)){	
		load<DTYPE, 3>::col(regC, 'x', srcL + glbRPos);
		load<DTYPE, 3>::col(regC, 'y', srcC + glbRPos);
		load<DTYPE, 3>::col(regC, 'z', srcR + glbRPos);
		load<DTYPE, 3>::col(regC, 'w', dstX + glbRPos);
	}
	else
	{
		load<DTYPE, 2>::col(regC, 'x', srcL + glbRPos);
		load<DTYPE, 2>::col(regC, 'y', srcC + glbRPos);
		load<DTYPE, 2>::col(regC, 'z', srcR + glbRPos);
		load<DTYPE, 2>::col(regC, 'w', dstX + glbRPos);
		regC[2]=regC[1];
	}


	Eqn<DTYPE> eqUpInit = regC[0];

	radix<RAD>(regC);	

	// The first radix stage is always executed
	const int MIXRAD = MIXR<N, RAD>::val;
	int num_threads= blockDim.x;	
	
	const int warpSize=(N>32)?32:N/RAD;
	const int warpAttack = warpSize;


	
	//shuffling reduction
	int i,j;
	for(i=1, j=1; i < (warpSize/2) ; i*=2,j++)		
	{

		Eqn<DTYPE> EqUp = shfl<DTYPE>::shfl_Eq_Up(regC[1],i,warpSize);
		Eqn<DTYPE> EqDown =shfl<DTYPE>::shfl_Eq_Down(regC[1],i,warpSize);
		if(isMultipleOf(j,threadId+1))
		{
			Eqn<DTYPE> eq1 = reduce(EqUp,regC[1],2);
	 		Eqn<DTYPE> eqR = reduce(eq1,EqDown, 0);

	 		regC[1]=eqR;

																	
		}
	}

		
		
		//exchanging

		Eqn<DTYPE> EqUp = shfl<DTYPE>::shfl_Eq_Up(regC[1],i,warpSize);
		Eqn<DTYPE> EqDown = shfl<DTYPE>::shfl_Eq_Down(regC[1],i,warpSize);

		DTYPE x = 0;
			if(threadId==(warpSize/2 -1))
			{	
				Eqn<DTYPE> eqA, eqB;
				eqA=regC[1];
				eqB=EqDown;
				DTYPE tmp = (eqB.y*eqA.y)-(eqA.z*eqB.x);		
				x = (eqB.y*eqA.w - eqA.z*eqB.w)/tmp;	
			}
			if(threadId==(warpSize-1))
			{
				Eqn<DTYPE> eqA, eqB;	
				eqB = regC[1];
				eqA = EqUp;

				DTYPE tmp = (eqB.y*eqA.y)-(eqA.z*eqB.x);
				x = (eqB.w*eqA.y - eqA.w*eqB.x)/tmp;	
			}
			

		//shuffling substitution
		for(i=warpSize/4;i>0;i/=2)
		{
			j--;

			DTYPE xUp = shfl<DTYPE>::shfl_Up(x,i,warpSize);
			DTYPE xDown =shfl<DTYPE>::shfl_Down(x,i,warpSize);

	
			if(isMultipleOf(j,threadId-(i-1)))
			{
				//Substitution
				Eqn<DTYPE> eq = regC[1];
				x= (eq.w - eq.x*xUp - eq.z*xDown)/eq.y;
			}
	
		}


		Eqn<DTYPE> eq = eqUpInit;
		DTYPE xUp = shfl<DTYPE>::shfl_Up(x,1,warpSize);
		DTYPE x2 = (eq.w -eq.x*xUp - eq.z*x)/eq.y;
		X[shmOffset+((N/warpAttack))*(threadId+1)-2]=x2;	
		
		X[shmOffset+((N/warpAttack))*(threadId+1)-1]=x;


	
	__syncthreads();
	
	#pragma unroll
	for(int i=0;i<(RAD);i++)
	{	
		dstX[glbWPos + threadId+ i * glbWStr]= X[shmOffset+threadId+i*glbWStr];
	}
	
}





// X is in SHM, N is larger than 64
template<class DTYPE, int N, int DIR, int RAD, int SHM> __global__ void
KTridiagCR4(const DTYPE* __restrict__ srcL, 
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
	int glbWPos = verticalId * stride ;
	int glbWStr = N / RAD;

	// Statically allocate registers and shared memory
	Eqn<DTYPE> regC[3];
	
	__shared__ Eqn<DTYPE> shm[N > RAD ? SHM : 1];
	__shared__ DTYPE X[N> RAD ? SHM : 1];

	load<DTYPE, RAD>::col(regC, 'x', srcL + glbRPos);
	load<DTYPE, RAD>::col(regC, 'y', srcC + glbRPos);
	load<DTYPE, RAD>::col(regC, 'z', srcR + glbRPos);
	load<DTYPE, RAD>::col(regC, 'w', dstX + glbRPos);
	
	copy<RAD>(shm+RAD*threadId+shmOffset,1,regC);
	__syncthreads();

	if(threadIdx.x!=(blockDim.x-1))
		copy<1>(&regC[2],&shm[shmOffset+RAD*threadId+2],0);//No last thread
	else regC[2]=regC[1];

	radix<RAD>(regC);	

	// The first radix stage is always executed
	const int MIXRAD = MIXR<N, RAD>::val;
	int num_threads= blockDim.x;
	
	
	const int warpSize=32;
	const int warpAttack = warpSize;

	#pragma unroll
	for(int accRad = MIXRAD; accRad < (N/warpAttack);  accRad *= RAD) {

		int cont = accRad;

		int strideW = cont;
		int indexW = strideW*threadId + strideW -1+shmOffset;

		int strideR = 2*cont;
		int indexR = strideR * threadId + strideR-1+shmOffset;

		if(cont > 1) __syncthreads();
	
		if(threadId<num_threads)
			copy<1>(shm+indexW, 0, &regC[1]);

		__syncthreads();
		
		num_threads/=2;
		
		if(threadId<num_threads)						
			copy<3>(regC,shm+indexR-cont,cont,((indexR-shmOffset+cont)>=N));
		
		radix<RAD>(regC);

	}
	

	if (threadId<warpSize)
	{
		int i,j;
		for(i=1, j=1; i < (warpSize/2) ; i*=2,j++)		
		{
			
			Eqn<DTYPE> EqUp = shfl<DTYPE>::shfl_Eq_Up(regC[1],i,warpSize);
			Eqn<DTYPE> EqDown =shfl<DTYPE>::shfl_Eq_Down(regC[1],i,warpSize);
			if(isMultipleOf(j,threadId+1))
			{
				Eqn<DTYPE> eq1 = reduce(EqUp,regC[1],2);
		 		Eqn<DTYPE> eqR = reduce(eq1,EqDown, 0);

		 		regC[1]=eqR;

																		
			}
		}

		Eqn<DTYPE> EqUp = shfl<DTYPE>::shfl_Eq_Up(regC[1],i,warpSize);
		Eqn<DTYPE> EqDown = shfl<DTYPE>::shfl_Eq_Down(regC[1],i,warpSize);

		DTYPE x = 0;
		
	
		if(threadId==(warpSize/2 -1))
			{	
				Eqn<DTYPE> eqA, eqB;
				eqA=regC[1];
				eqB=EqDown;
				DTYPE tmp = (eqB.y*eqA.y)-(eqA.z*eqB.x);		
				x = (eqB.y*eqA.w - eqA.z*eqB.w)/tmp;	
	
			}
		if(threadId==(warpSize-1))
			{
				Eqn<DTYPE> eqA, eqB;	
				eqB = regC[1];
				eqA = EqUp;

				DTYPE tmp = (eqB.y*eqA.y)-(eqA.z*eqB.x);
				x = (eqB.w*eqA.y - eqA.w*eqB.x)/tmp;	
			}
			

		
		//shuffling substitution
		for(i=warpSize/4;i>0;i/=2)
		{
			j--;

			DTYPE xUp = shfl<DTYPE>::shfl_Up(x,i,warpSize);
			DTYPE xDown =shfl<DTYPE>::shfl_Down(x,i,warpSize);

			if(isMultipleOf(j,threadId-(i-1)))
			{
				//substitution
				Eqn<DTYPE> eq = regC[1];
				x= (eq.w - eq.x*xUp - eq.z*xDown)/eq.y;
			}
	
		}

		X[shmOffset+((N/warpAttack))*(threadId+1)-1]=x;


	}

	__syncthreads();

	num_threads=warpSize;
	
	
	#pragma unroll	
	for (int j = N/warpAttack; j > 1; j/=2)
   	{	

	       int delta = j/2;

	       __syncthreads();
	       if (threadId < num_threads)
	       {
		   int d = shmOffset;
		   int i = j * threadId + j/2 - 1;
		   Eqn<DTYPE> eq;
		   copy<1>(&eq,shm+shmOffset+i,1);
		 
		   if(i == delta - 1)
			X[d+i]= (eq.w - eq.z*X[d+i+delta])/eq.y;
			 			
		   else
		  	X[d+i]=	(eq.w - eq.x*X[d+i-delta] - eq.z*X[d+i+delta])/eq.y;
		}
		num_threads *= 2;
     	}


	__syncthreads();
	
	#pragma unroll
	for(int i=0;i<(RAD);i++)
	{	
		dstX[glbWPos + threadId+ i * glbWStr]= X[shmOffset+threadId+i*glbWStr];
	}
}



// --- BranchTable -------------------------------------------------------



const static kernelCfg<float> triXf32A[] = { //! GPU dependent
	NULL_ROW(1),
	NULL_ROW(2),
	ROW(KTridiagCR2, float,    4, 128, 2),
	ROW(KTridiagCR2, float,    8, 256, 2),
	ROW(KTridiagCR2, float,   16, 256, 2),
	ROW(KTridiagCR2, float,   32, 256, 2),
	ROW(KTridiagCR2, float,   64, 256, 2),
	ROW(KTridiagCR, float,  128, 256, 2),
	ROW(KTridiagCR, float,  256, 256, 2),
	ROW(KTridiagCR, float,  512, 512, 2),
	ROW(KTridiagCR, float, 1024,1024, 2),
	NULL_ROW(4096),
};



//Maxwell configuration

const static kernelCfg<float> triXf32B[] = { //! GPU dependent
	NULL_ROW(1),
	NULL_ROW(2),
	ROW(KTridiagCR3, float,    4, 128, 2),
	ROW(KTridiagCR3, float,    8, 256, 2),
	ROW(KTridiagCR3, float,   16, 256, 2),
	ROW(KTridiagCR3, float,   32, 256, 2),
	ROW(KTridiagCR3, float,   64, 128, 2),
	ROW(KTridiagCR4, float,  128, 128, 2),
	ROW(KTridiagCR , float,  256, 256, 2),
	ROW(KTridiagCR , float,  512, 512, 2),
	ROW(KTridiagCR4, float, 1024,1024, 2),
	NULL_ROW(4096),
};









//---- Interface Functions -----------------------------------------------

//- Main library function for 'float' equations
int KTridiagCR(float* data, int dir, int N, int M, int batch) {

	if(N>1024)
		return -1;
	return KLauncher3(tabla, sizeof(tabla), data, dir, N, batch);
}


