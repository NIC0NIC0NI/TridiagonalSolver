//- =======================================================================
//+ BPLG BMCS Sort algorithm 
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "tools/cudacl.hxx"
#include "inc/op_copy.hxx"
#include "inc/op_twiddle.hxx"
#include "inc/op_load.hxx"
#include "KLauncher1.hxx"
//---- Include Section ---------------------------------------------------
#include <cstdio>


#ifndef __CUDA_ARCH__
    #define __CUDA_ARCH__ CUDA_ARCH
#endif


#if  __CUDA_ARCH__ < 400
	#define tabla1 triXiA
#endif

#if  __CUDA_ARCH__ >= 400
	#define tabla1 triXiB
#endif



//---- Butterfly operator ------------------------------------------------

template<class DTYPE, int size> struct butterflyB {
	static inline __device__ void
	init(DTYPE * data, int s = 1);

	
	//- Generic butterfly step, is more efficient to call 'butterfly_init' first
	static inline __device__ void
	step(DTYPE * data, int s = 1);

};


template<class DTYPE> struct butterflyB<DTYPE, 4> {
	
	static inline __device__ void
	init(DTYPE * data, int s = 1) {
		if(data[0]>data[1]) {
			DTYPE aux = data[0];
			data[0]=data[1];
			data[1]=aux;	
		}
		
		if(data[2]>data[3]) {
			DTYPE aux = data[2];
			data[2]=data[3];
			data[3]=aux;	
		}
		
		if(data[0]>data[3]) {
			DTYPE aux = data[0];
			data[0]=data[3];
			data[3]=aux;	
		}
	
		if(data[1]>data[2]) {
			DTYPE aux = data[1];
			data[1]=data[2];
			data[2]=aux;	
		}
		if(data[0]>data[1]) {
			DTYPE aux = data[0];
			data[0]=data[1];
			data[1]=aux;	
		}
		
		if(data[2]>data[3]) {
			DTYPE aux = data[2];
			data[2]=data[3];
			data[3]=aux;	
		}
		
	}

	
	//- The general Rad<2> case is defined according to the equation
	static inline __device__ void
	step(DTYPE * data, int s = 1) {
		
		if(data[0]>data[3]) {
			DTYPE aux = data[0];
			data[0]=data[3];
			data[3]=aux;	
		}
		
		if(data[1]>data[2]) {
			DTYPE aux = data[1];
			data[1]=data[2];
			data[2]=aux;	
		}
		
		
		
	}
};



template<class DTYPE, int size> struct butterfly {

	//- The first butterfly step is an optimized version of 'butterfly_step'
	static inline __device__ void
	init(DTYPE * data, int s = 1);

	//- Generic butterfly step, is more efficient to call 'butterfly_init' first
	static inline __device__ void
	step(DTYPE * data, int s = 1);

};

template<class DTYPE> struct butterfly<DTYPE, 2> {

	//- The initial Rad<2> case is optimized
	static inline __device__ void
	init(DTYPE * data, int s = 1) {
		if(data[0]>data[1]) {
			DTYPE aux = data[0];
			data[0]=data[1];
			data[1]=aux;	
		}
		
		
	}

	//- The general Rad<2> case is defined according to the equation
	static inline __device__ void
	step(DTYPE * data, int s = 1) {
		if(data[0]>data[1]) {
			DTYPE aux = data[0];
			data[0]=data[1];
			data[1]=aux;	
		}
	}
};


template<class DTYPE> struct butterfly<DTYPE, 4> {

	//- The initial Rad<4> case is optimized
	static inline __device__ void
	init(DTYPE * data, int s = 1) {
	if(data[0]>data[3]) {
		int aux = data[0];
		data[0]= data[3];
		data[3]=aux;	
	}
	if(data[1]>data[2]) {
		int aux = data[1];
		data[1]= data[2];
		data[2]=aux;	
	}
	if(data[0]>data[1]) {
		int aux = data[0];
		data[0]=data[1];
		data[1]=aux;	
	}
	if(data[2]>data[3]) {
		float aux = data[2];
		data[2]=data[3];
		data[3]=aux;	
	}

		
	}

	//- The general Rad<4> case is defined according to the equation
	static inline __device__ void
	step(DTYPE * data, int s = 1) {
		if(data[0]>data[2]) {
			int aux = data[0];
			data[0]=data[2];
			data[2]=aux;	
		}
		if(data[1]>data[3]) {
			int aux = data[1];
			data[1]=data[3];
			data[3]=aux;	
		}
		if(data[0]>data[1]) {
			int aux = data[0];
			data[0]=data[1];
			data[1]=aux;	
		}
		if(data[2]>data[3]) {
			int aux = data[2];
			data[2]= data[3];
			data[3] =aux;	
		}

	}
};

template<class DTYPE> struct butterfly<DTYPE, 8> {

	//- The initial Rad<8> case is recursively defined
	static inline __device__ void
	init(DTYPE * data, int s = 1) {
		
	}

	//- The general Rad<8> case is recursively defined
	static inline __device__ void
	step(DTYPE * data, int s = 1) {
	
	
	}
};


//---- Radix operator ----------------------------------------------------

template<int RAD, class DTYPE> inline __device__ void
radixB(DTYPE* data) {
	butterflyB<DTYPE, RAD>::step(data);
}


//- Mixed-radix stage, only called once before the main loop
template<int SIZE, int RAD, class DTYPE> inline __device__ void
radixB(DTYPE * data) {
	#pragma unroll
	for(int i = 0; i < SIZE; i+= RAD)
		butterflyB<DTYPE, RAD>::init(data+i);
}


//- Generic radix stage, used in the main loop of the algorithm
template<int RAD, class DTYPE> inline __device__ void
radix(DTYPE* data) {
	butterfly<DTYPE, RAD>::step(data);
}

//- Mixed-radix stage, only called once before the main loop
template<int SIZE, int RAD, class DTYPE> inline __device__ void
radix(DTYPE * data) {
	#pragma unroll
	for(int i = 0; i < SIZE; i+= RAD)
		butterfly<DTYPE, RAD>::init(data+i);
}

//---- Global Memory Load ------------------------------------------------




//------- Cuda Kernels ---------------------------------------------------

__device__ inline int ide1 (int k)
{
	const int modulo = (threadIdx.x & ((k/2)-1));
	return 2*k*(threadIdx.x/(k/2)) + modulo;
}
__device__ inline int ide2 (int k, int id1) 
{
	return id1 + (2*k-1-2*(id1&(k-1))); 
}

__device__ inline void compara8b (int* elems1, int* elems2,int marca)
{

	 if((!marca)?(elems1[0]>elems2[0]):(elems1[0]<elems2[0]))
	 {
			int aux = elems1[0];
			elems1[0]=elems2[0];
			elems2[0]=aux;	
	 	
	 }

	if((!marca)?(elems1[1]>elems2[1]):(elems1[1]<elems2[1]))
	 {
			int aux = elems1[1];
			elems1[1]=elems2[1];
			elems2[1]=aux;	
	 	
	 }
	 
	 if((!marca)?(elems1[2]>elems2[2]):(elems1[2]<elems2[2]))
	 {
			int aux = elems1[2];
			elems1[2]=elems2[2];
			elems2[2]=aux;	
	 	
	 }
	 
	 if((!marca)?(elems1[3]>elems2[3]):(elems1[3]<elems2[3]))
	 {
			int aux = elems1[3];
			elems1[3]=elems2[3];
			elems2[3]=aux;	
	 	
	 }
	 
 }



__device__ inline void compara8 (int* elems1, int* elems2,int marca)
{

	 if((!marca)?(elems1[0]>elems2[3]):(elems1[0]<elems2[3]))
	 {
			int aux = elems1[0];
			elems1[0]=elems2[3];
			elems2[3]=aux;	
	 	
	 }

	if((!marca)?(elems1[1]>elems2[2]):(elems1[1]<elems2[2]))
	 {
			int aux = elems1[1];
			elems1[1]=elems2[2];
			elems2[2]=aux;	
	 	
	 }
	 
	 if((!marca)?(elems1[2]>elems2[1]):(elems1[2]<elems2[1]))
	 {
			int aux = elems1[2];
			elems1[2]=elems2[1];
			elems2[1]=aux;	
	 	
	 }
	 
	 if((!marca)?(elems1[3]>elems2[0]):(elems1[3]<elems2[0]))
	 {
			int aux = elems1[3];
			elems1[3]=elems2[0];
			elems2[0]=aux;	
	 	
	 }



}





template< int N, int DIR, int RAD, int SHM> __global__ void
KSort2(int4 *  data, int stride)
{
	// Obtain group-1D, thread-X and batch-Y identifiers
	int groupId = get_group_id(0) + get_group_id(1) * get_num_groups(0);
	int threadId = get_local_id(0);      // Thread horizontal (N)
	int batchId = get_local_id(1);       // Thread vertical (batch)
	int thread1D = threadId + batchId * N/4;
	
	int glbRPos =(groupId * SHM)/4 + thread1D ;
	// Offset for accesing thread data
	int shmOffset = batchId * N;

	
	

	// Statically allocate registers and shared memory
	int reg[RAD];
	__shared__ int shm[N > RAD ? SHM : 1];

	
	int4 regs;	
	regs=data[glbRPos];	
	reg[0]=regs.x; reg[1]=regs.y;reg[2] =regs.z; reg[3]=regs.w;
	
	
	
	radixB<4,4>(reg);
	
	
	
	
	bool mask = false;
	// Process the remaining stages
	
	//Using shuffle for warp sizes
	#pragma unroll
	for(int cont = 1, accRad = 4; accRad < ((N<128)?N:128); cont = accRad, accRad *= 2) {
	
		
		
		const int width = accRad/2;
		const int laneId = threadId & (width-1);
		int aux[4];
	
		aux[0] = __shfl(reg[0], width-1-laneId, width);
		aux[1] = __shfl(reg[1], width-1-laneId, width);
		aux[2] = __shfl(reg[2], width-1-laneId, width);
		aux[3] = __shfl(reg[3], width-1-laneId, width);
	
	
		compara8(reg ,aux, laneId>=(width/2) );
	
		for (int j=accRad/2; j>2; j/=2)
		{
				const int width = j/2;
				const int laneId = threadId & (width-1); 
			
				aux[0]= __shfl(reg[0], (laneId<(width/2))?(laneId+j/4):(laneId -j/4), width);
				aux[1]= __shfl(reg[1], (laneId<(width/2))?(laneId+j/4):(laneId -j/4), width);
				aux[2]= __shfl(reg[2], (laneId<(width/2))?(laneId+j/4):(laneId -j/4), width);
				aux[3]= __shfl(reg[3], (laneId<(width/2))?(laneId+j/4):(laneId -j/4), width);						
			
				compara8b(reg ,aux, laneId>=(width/2) );
		}
			
			
			radix<4>(reg);
			mask= !mask;
	
	}
	
	
	
	
	//Using shared memory
	#pragma unroll
	for(int accRad = 128; accRad < N;  accRad *= 2) {
		
		
		
		const int id1 = ide1(accRad);
		const int id2 = id1+accRad/2;
		const int id3 = ide2(accRad,id2); 
		const int id4 = ide2(accRad,id1); 
		
		
		if(accRad > 128) __syncthreads();
		
		copy<RAD>(shm+shmOffset+RAD*threadId,reg,1);
		
		__syncthreads();
		
		copy<1>(reg,1,shm+shmOffset+id1,1);
		copy<1>(reg+1,1,shm+shmOffset+id2,1);
		copy<1>(reg+2,1,shm+shmOffset+id3,1);
		copy<1>(reg+3,1,shm+shmOffset+id4,1);
		
		if(mask)		
			radix<4,4>(reg);
		else 
			radixB<4>(reg);
				
		int el1 = id1;
		int el2 = id2;
		int el3 = id3;
		int el4 = id4;
		
		for(int j=((mask)?(accRad/2):accRad);j>1;j/=4)
		{
			
			
			if (j<(accRad/2)) __syncthreads();
			
			copy<1>(shm+shmOffset+el1,1,reg,1);
			copy<1>(shm+shmOffset+el2,1,reg+1,1);
			copy<1>(shm+shmOffset+el3,1,reg+2,1);
			copy<1>(shm+shmOffset+el4,1,reg+3,1);
			
			__syncthreads();
			
			el1 = j*(threadId/(j/4))+(threadId&(j/4-1));
			el2 = el1+j/4; 
			el3 = el1+j/2;
			el4 = el2+j/2;
			
			
			copy<1>(reg,1,shm+shmOffset+el1,1);
			copy<1>(reg+1,1,shm+shmOffset+el2,1);
			copy<1>(reg+2,1,shm+shmOffset+el3,1);
			copy<1>(reg+3,1,shm+shmOffset+el4,1);
			
			
			radix<4>(reg);			
			
			
		}
		mask= !mask;
			}
	
	
	
	
	regs.x=reg[0]; regs.y=reg[1];regs.z=reg[2]; regs.w=reg[3];
	
	data[glbRPos]=regs;


}


// --- BranchTable -------------------------------------------------------






const static kernelCfg triXiA[] = { //! GPU dependent
	NULL_ROW(1),
	NULL_ROW(2),
	ROW(KSort2,    4, 256, 4),
	ROW(KSort2,    8, 256, 4),
	ROW(KSort2,   16, 256, 4),
	ROW(KSort2,   32, 256, 4),
	ROW(KSort2,   64, 256, 4),
	ROW(KSort2,  128, 256, 4),
	ROW(KSort2,  256, 256, 4),
	ROW(KSort2,  512, 512, 4), 
	ROW(KSort2, 1024,1024, 4),
	ROW(KSort2, 2048,2048, 4),
	ROW(KSort2, 4096,4096, 4),
	NULL_ROW(4096),
};

const static kernelCfg triXiB[] = { //! GPU dependent
	NULL_ROW(1),
	NULL_ROW(2),
	ROW(KSort2,    4, 256, 4),
	ROW(KSort2,    8, 256, 4),
	ROW(KSort2,    16, 256, 4),
	ROW(KSort2,    32, 256, 4),
	ROW(KSort2,    64, 256, 4),
	ROW(KSort2,   128, 256, 4),
	ROW(KSort2,   256, 256, 4),
	ROW(KSort2,   512, 512, 4), 
	ROW(KSort2,  1024,1024, 4),
	ROW(KSort2,  2048,2048, 4),
	ROW(KSort2,  4096,4096, 4),
	NULL_ROW(4096),
};



// --- Launcher ----------------------------------------------------------




//---- Interface Functions -----------------------------------------------


int KSort(int* data, int dir, int N, int M, int batch){
	
	if(N>4096)
		return -1;

	return  KLauncher1(tabla1, sizeof(tabla1), data, dir, N, batch);
}




