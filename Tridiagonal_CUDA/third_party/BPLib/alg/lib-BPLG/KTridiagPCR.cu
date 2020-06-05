//- =======================================================================
//+ KTridiag PCR algorithm 
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "tools/cudacl.hxx"
#include "inc/op_copy.hxx"
#include "inc/op_twiddle.hxx"
#include "KLauncher3.hxx"

#include "inc/op_mixr.hxx"
#include "inc/op_reduce.hxx"
#include "inc/op_load.hxx"



#ifndef __CUDA_ARCH__
    #define __CUDA_ARCH__ CUDA_ARCH
#endif


#if  __CUDA_ARCH__ < 400
	#define tabla1 triXf32A
#endif

#if  __CUDA_ARCH__ >= 400
	#define tabla1 triXf32B
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

//------- Cuda Kernels ---------------------------------------------------

 // RAD 2 implementation reducing the number of registers, X in SHM

template<class DTYPE, int N, int DIR, int RAD, int SHM> __global__ void
KTridiagPCR3(const DTYPE* __restrict__ srcL, 
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

	int glbRPos = verticalId * stride + threadId ;
	int glbWPos = verticalId * stride + threadId;
	int glbWStr = N / RAD;

	// Statically allocate registers and shared memory
	Eqn<DTYPE> regC[6];
	__shared__ Eqn<DTYPE> shm[N > RAD ? SHM : 1];
	__shared__ DTYPE X[N> RAD ? SHM : 1];

				
		load<DTYPE, 1>::col(&(regC[1]), 'x', srcL + glbRPos);
		load<DTYPE, 1>::col(&(regC[1]), 'y', srcC + glbRPos);
		load<DTYPE, 1>::col(&(regC[1]), 'z', srcR + glbRPos);
		load<DTYPE, 1>::col(&(regC[1]), 'w', dstX + glbRPos);
		

		load<DTYPE, 1>::col(&(regC[4]), 'x', srcL + glbRPos+blockDim.x);
		load<DTYPE, 1>::col(&(regC[4]), 'y', srcC + glbRPos+blockDim.x);
		load<DTYPE, 1>::col(&(regC[4]), 'z', srcR + glbRPos+blockDim.x);
		load<DTYPE, 1>::col(&(regC[4]), 'w', dstX + glbRPos+blockDim.x);
 	

	// The first radix stage is always executed
	const int MIXRAD = 2;
	int num_threads= blockDim.x;
	int cont = 1;

	
	copy<RAD>(shm+threadId+shmOffset,blockDim.x,regC+1,3);
	
	__syncthreads();

		copy<1,false,RAD>(&(regC[2]),shm+shmOffset+threadId+1,cont); 
		copy<1,true,RAD> (regC,shm+shmOffset+threadId-1,cont);
 	
		copy<1,false,RAD>(&(regC[5]),shm+shmOffset+threadId+1+blockDim.x,cont); 
		copy<1,true,RAD> (&(regC[3]),shm+shmOffset+threadId-1+blockDim.x,cont); 


	
		if((threadIdx.x)<cont)
			regC[0]=regC[1];
		if((threadIdx.x)>=(N-cont))	
			regC[2]=regC[1];

		if((threadIdx.x+blockDim.x)<cont)
			regC[3]=regC[4];
		if((threadIdx.x+blockDim.x)>=(N-cont))	
			regC[5]=regC[4];
	


		radix<2>(regC);	
		radix<2>(&(regC[3]));	
	
	

	
	

	

	// Process the remaining stages
	#pragma unroll
	for(int accRad = MIXRAD; accRad < (N/2);  accRad *= 2) {

		

		cont=accRad;
		

		if(cont > 1) __syncthreads(); 
	
			
		copy<1>(shm+threadId+shmOffset, 0, regC+1);
		copy<1>(shm+threadId+shmOffset+blockDim.x, 0, regC+4);		


		__syncthreads(); 
		
		
		
		num_threads/=2;
		
		
			copy<1>(regC+1,shm+shmOffset+threadId,1);		
			copy<1,false,RAD>(regC+2,shm+shmOffset+threadId+cont,cont); 
			copy<1,true,RAD>(regC,shm+shmOffset+threadId-cont,cont);
			copy<1>(regC+4,shm+shmOffset+threadId+blockDim.x,1);		
			copy<1,false,RAD>(regC+5,shm+shmOffset+threadId+cont+blockDim.x,cont); 
			copy<1,true,RAD>(regC+3,shm+shmOffset+threadId-cont+blockDim.x,cont);



			if((threadIdx.x)<cont)
				regC[0]=regC[1];
			if((threadIdx.x)>=(N-cont))	
				regC[2]=regC[1];

			if((threadIdx.x+blockDim.x)<cont)
				regC[03]=regC[4];
			if((threadIdx.x+blockDim.x)>=(N-cont))	
				regC[5]=regC[4];
		
			radix<2>(regC);
			radix<2>(&(regC[3]));
			
	}

	
	
		copy<1>(shm+shmOffset+threadId,0,regC+1);
		copy<1>(shm+shmOffset+threadId+blockDim.x,0,regC+4);
	
	__syncthreads();

	
		if((threadId)<(N/2))
		{
			Eqn<DTYPE> eq1,eq2;
			eq1=regC[1];
			copy<1>(&eq2,shm+shmOffset+threadId+N/2,1);
			DTYPE tmp = eq2.y*eq1.y - eq1.z*eq2.x;
			X[shmOffset+threadId]= (eq2.y*eq1.w - eq1.z*eq2.w)/tmp;
			X[shmOffset+threadId+N/2]= (eq2.w*eq1.y - eq1.w*eq2.x)/tmp;		
		
		}

		if((threadId+blockDim.x)<(N/2))
		{
			Eqn<DTYPE> eq1,eq2;
			eq1=regC[4];
			copy<1>(&eq2,shm+shmOffset+threadId+blockDim.x+N/2,1);
			DTYPE tmp = eq2.y*eq1.y - eq1.z*eq2.x;
			X[shmOffset+threadId+blockDim.x]= (eq2.y*eq1.w - eq1.z*eq2.w)/tmp;
			X[shmOffset+threadId+N/2+blockDim.x]= (eq2.w*eq1.y - eq1.w*eq2.x)/tmp;		
		
		}
	

	//Substitution

	__syncthreads();
	

	#pragma unroll
	for(int i=0;i<(RAD);i++)
	{	
		dstX[glbWPos + i * glbWStr]= X[shmOffset+threadId+i*glbWStr];
	}	
	
}






// X is in GLOBAL MEMORY

template<class DTYPE, int N, int DIR, int RAD, int SHM> __global__ void
KTridiagPCR2(const DTYPE* __restrict__ srcL, 
		 const DTYPE* __restrict__ srcC,
		 const DTYPE* __restrict__ srcR,
		       DTYPE* dstX, int stride)
{
	int groupId = get_group_id(0) + get_group_id(1) * get_num_groups(0);
	int threadId = get_local_id(0);      // Thread horizontal (N)
	int batchId = get_local_id(1);       // Thread vertical (batch)
	int verticalId = groupId * get_local_size(1) + batchId;

	// Offset for accesing thread data
	int shmOffset = batchId * N;

	int glbRPos = verticalId * stride + 2* threadId  ;
	int glbWPos = verticalId * stride + threadId;
	int glbWStr = N / 2;

	// Statically allocate registers and shared memory
	Eqn<DTYPE> regC[4];

	
	__shared__ Eqn<DTYPE> shm[N > 2 ? SHM : 1];

	
	// Load 'regC'. Left and right equations are initialized by 'radix_init'
	
	load<DTYPE, 2>::col(&regC[1], 'x', srcL + glbRPos);
	load<DTYPE, 2>::col(&regC[1], 'y', srcC + glbRPos);
	load<DTYPE, 2>::col(&regC[1], 'z', srcR + glbRPos);
	load<DTYPE, 2>::col(&regC[1], 'w', dstX + glbRPos);
		

	
	int cont = 1;
	copy<2>(shm + 2*threadId+shmOffset,1,&regC[1]);

	__syncthreads();
	


	//0 bellow, 1 above
	copy2<1>(&regC[3],shm+shmOffset+2*threadId+2,false,cont);
	copy2<1>(&regC[0],shm+shmOffset+2*threadId-1,true,cont); 





	if(threadIdx.x<1)
		regC[0]=regC[1];
	if(threadIdx.x>=((N/2)-1))	
		regC[3]=regC[2];


	
	Eqn<DTYPE>aux[3];
	aux[0]=regC[0]; aux[1]=regC[1]; aux[2]=regC[2];
	
	radix<2>(aux);
	radix<2>(regC+1);
	regC[1]=aux[1];	



	// Process the remaining stages
	#pragma unroll
	for(int accRad = 2; accRad < (N/2);  accRad *= 2) {
	

		const int writeIndex = (threadId/cont)*accRad + (threadId&(cont-1));
		const int strW = cont ;

		const int readIndex = (threadId/accRad)*2*accRad + (threadId&(accRad-1)) ;
		const int strR = accRad;	
		
	
		if(cont > 1) __syncthreads(); 
	
		copy<2>(shm+shmOffset+writeIndex, strW, &regC[1]);

		
		
		__syncthreads();
		copy<2>(&regC[1],shm+shmOffset+readIndex,strR);
		
		copy2<1>(&regC[3],shm+shmOffset+readIndex+2*accRad,false,accRad); 
		copy2<1>(&regC[0],shm+shmOffset+readIndex-accRad,true,accRad);
	
		

		if(threadIdx.x<accRad)
			regC[0]=regC[1];
	
		if(threadIdx.x>=(N/2-accRad))
			regC[3]=regC[2];

	
		Eqn<DTYPE>aux[3];
		aux[0]=regC[0]; aux[1]=regC[1]; aux[2]=regC[2];
		
		radix<2>(aux);
		radix<2>(regC+1);
		regC[1]=aux[1];	

		cont=accRad;
	}
	int writeIndex = (threadId/cont)*(N/2)+(threadId&(cont-1));
	

	copy<2>(shm+shmOffset+writeIndex,cont,&regC[1]);
	
	__syncthreads();



	if(threadId< (N/4)){

		Eqn<DTYPE> eq1,eq2;
		eq1=regC[1];
		copy<1>(&eq2,shm+shmOffset+threadId+N/2,1);	
				
		DTYPE tmp = eq2.y*eq1.y - eq1.z*eq2.x;
		dstX[glbWPos]= (eq2.y*eq1.w - eq1.z*eq2.w)/tmp;
		dstX[glbWPos+N/2]= (eq2.w*eq1.y - eq1.w*eq2.x)/tmp;
		
		eq1=regC[2];
		copy<1>(&eq2,shm+shmOffset+threadId+N/2+N/4,1);
		tmp = 	eq2.y*eq1.y - eq1.z*eq2.x;	
		dstX[glbWPos+N/4]= (eq2.y*eq1.w - eq1.z*eq2.w)/tmp;
		dstX[glbWPos+N/2+N/4]= (eq2.w*eq1.y - eq1.w*eq2.x)/tmp;
	}	

	

}



template<class DTYPE, int N, int DIR, int RAD, int SHM> __global__ void
KTridiagPCR(const DTYPE* __restrict__ srcL, 
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
	Eqn<DTYPE> regC[3];
	
	__shared__ Eqn<DTYPE> shm[N > RAD ? SHM : 1];

	// Load 'regC'. Left and right equations are initialized by 'radix_init'
	
	load<DTYPE, RAD>::col(&regC[1], 'x', srcL + glbRPos);
	load<DTYPE, RAD>::col(&regC[1], 'y', srcC + glbRPos);
	load<DTYPE, RAD>::col(&regC[1], 'z', srcR + glbRPos);
	load<DTYPE, RAD>::col(&regC[1], 'w', dstX + glbRPos);

	// The first radix stage is always executed
	const int MIXRAD = 2;
	int num_threads= blockDim.x;
	int cont = 1;

	

	copy<RAD>(shm+RAD*threadId+shmOffset,1,&regC[1]);

	__syncthreads();

	copy2<1>(&regC[2],shm+shmOffset+RAD*threadId+1,false,cont); 
	copy2<1>(&regC[0],shm+shmOffset+RAD*threadId-1,true,cont); 


	if(threadIdx.x<cont)
		regC[0]=regC[1];
	if(threadIdx.x>=(N-cont))	
		regC[2]=regC[1];


	


	radix<2>(regC);	
	
	// Process the remaining stages
	#pragma unroll
	for(int accRad = MIXRAD; accRad < (N/2);  accRad *= 2) {

		cont=accRad;		

		if(cont > 1) __syncthreads(); 
	
		//if(threadId<num_threads)
			copy<1>(shm+threadId+shmOffset, 0, &regC[1]);

		
		
		__syncthreads(); 
		num_threads/=2;
		
		copy<1>(&regC[1],shm+shmOffset+threadId,1);
		
		copy2<1>(&regC[2],shm+shmOffset+RAD*threadId+cont,false,cont); 
		copy2<1>(&regC[0],shm+shmOffset+RAD*threadId-cont,true,cont);

		if(threadIdx.x<cont)
			regC[0]=regC[1];
		if(threadIdx.x>=(N-cont))	
			regC[2]=regC[1];
		
		radix<2>(regC);

	}


	

	copy<1>(shm+shmOffset+threadId,0,&regC[1]);
	
	__syncthreads();

	

	if(threadId<(N/2))
	{
		Eqn<DTYPE> eq1,eq2;
		eq1=regC[1];
		copy<1>(&eq2,shm+shmOffset+threadId+N/2,1);
		DTYPE tmp = eq2.y*eq1.y - eq1.z*eq2.x;
		dstX[glbWPos]= (eq2.y*eq1.w - eq1.z*eq2.w)/tmp;
		dstX[glbWPos + N/2]= (eq2.w*eq1.y - eq1.w*eq2.x)/tmp;		
		
	}


	
}




// --- BranchTable -------------------------------------------------------


//- Template instantiation and branchtable for 'float' kernels in Kepler Architecture
const static kernelCfg<float> triXf32A[] = { //! GPU dependent
	NULL_ROW(1),
	NULL_ROW(2),
	ROW(KTridiagPCR, float,    4, 256, 1),
	ROW(KTridiagPCR, float,    8, 256, 1),
	ROW(KTridiagPCR, float,   16, 256, 1),
	ROW(KTridiagPCR, float,   32, 256, 1),
	ROW(KTridiagPCR, float,   64, 256, 1),
	ROW(KTridiagPCR, float,  128, 256, 1),
	ROW(KTridiagPCR, float,  256, 256, 1),
	ROW(KTridiagPCR2, float,  512, 512, 2),
	ROW(KTridiagPCR2, float, 1024, 1024, 2),//
	NULL_ROW(4096),
};


const static kernelCfg<float> triXf32B[] = { //! GPU dependent
	NULL_ROW(1),
	NULL_ROW(2),
	ROW(KTridiagPCR3, float,    4, 256, 2),
	ROW(KTridiagPCR3, float,    8, 256, 2),
	ROW(KTridiagPCR3, float,   16, 256, 2),
	ROW(KTridiagPCR3, float,   32, 256, 2),
	ROW(KTridiagPCR3, float,   64, 256, 2),
	ROW(KTridiagPCR3, float,  128, 128, 2),
	ROW(KTridiagPCR3, float,  256, 256, 2),
	ROW(KTridiagPCR3, float,  512, 512, 2),
	ROW(KTridiagPCR3, float, 1024, 1024, 2),//
	NULL_ROW(4096),
};







int KTridiagPCR(float* data, int dir, int N, int M, int batch) {

	if(N>1024)
		return -1;

	
	return KLauncher3(tabla1, sizeof(tabla1), data, dir, N, batch);
}


