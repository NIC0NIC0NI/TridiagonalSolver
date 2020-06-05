//- =======================================================================
//+ GPU Scan based on the BPLG library KS
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "tools/cudacl.hxx"
#include "inc/op_copy.hxx"
#include "inc/op_twiddle.hxx"
#include "KLauncher2.hxx"
#include "inc/op_shfl.hxx"

//---- Include Section ---------------------------------------------------
#include <cstdio>

#ifndef __CUDA_ARCH__
    #define __CUDA_ARCH__ CUDA_ARCH
#endif


#if  __CUDA_ARCH__ < 400
	#define tabla6 triXf32A
#endif

#if  __CUDA_ARCH__ >= 400
	#define tabla6 triXf32B
#endif


//---- Butterfly operator ------------------------------------------------

template<class DTYPE, int size> struct butterfly {

	//- Generic butterfly step, inclusive scan
	static inline __device__ void
	inc(DTYPE* data) {
		#pragma unroll
		for(int i = 1; i < size; i++)
			data[i] += data[i-1];
	}

	//- Generic butterfly step, exclusive scan
	static inline __device__ void
	exc(DTYPE* data){
		DTYPE acc = 0;
		#pragma unroll
		for(int i = 0; i < size; i++) {
			DTYPE tmp = data[i];
			data[i] = acc;
			acc += tmp;
		}
	}

};

//---- Radix operator ----------------------------------------------------

//- Normal radix stage, performs a normal inclusive scan
template<int RAD, class DTYPE> inline __device__ void
radix(DTYPE* data) {
	butterfly<DTYPE, RAD>::inc(data);
}

//- Mixed-radix stage, only called once before the main loop
template<int SIZE, int RAD, class DTYPE> inline __device__ void
radix(DTYPE* data) {
	#pragma unroll
	for(int i = 0; i < SIZE; i+= RAD)
		butterfly<DTYPE, RAD>::inc(data + i);
}

//- Generic radix stage, used in the main loop of the algorithm
template<int RAD, class DTYPE> inline __device__ void
radix(DTYPE* data, const DTYPE* rep, int stride, const DTYPE& base = 0) {
	DTYPE acc = base;
	#pragma unroll
	for(int i = 0; i < RAD; i++) {
		data[i] += acc;
		acc += rep[stride * i];
	}
}

//------- Cuda Kernels ---------------------------------------------------


template<int N, int DIR, int RAD, int SHM> __global__ void
KScanKS2(const float4* __restrict__ src, float4* dst, int stride)
{
	// Obtain group-1D, thread-X and batch-Y identifiers
	int groupId = get_group_id(0) + get_group_id(1) * get_num_groups(0);
	int threadId = get_local_id(0);      // Thread horizontal (N)
	int batchId = get_local_id(1);       // Thread vertical (batch)
	int thread1D = threadId + batchId * N/4;

	const int warp_size = 32;		
	// Offset for accessing thread data
	int shmOffset = batchId * warp_size;            // ShMem batch offset
       // Read Stride
	int glbRPos = (groupId * SHM)/4 + thread1D; // Read Pos
	
	// Statically allocate registers and shared memory
	float4 reg;
	
	__shared__ float shm[warp_size*(SHM/N)]; 
	reg = src[glbRPos];
	
	reg.y+=reg.x;
	reg.z+=reg.y;
	reg.w+=reg.z;
	



	
	int warp_id = threadIdx.x / warp_size;
	int lane_id = (threadId &(warp_size-1));


	
	float n1 = 0;

	#pragma unroll
	for(int i= 1; i<warp_size;i*=2)
	{
		
		float n = shfl<float>::shfl_Up(reg.w, i, warp_size);
		if(lane_id >= i){
			reg.w+=n;
			n1+=n;		
		}	
		

	}

	reg.x+=n1; reg.y+= n1; reg.z+=n1;

	

	if( (threadId & (warp_size-1))== (warp_size -1) )
		shm[warp_id+shmOffset]=reg.w;
	
	__syncthreads();
	
	
	if(!warp_id)
	{
		float warp_sum = 0;
				
		warp_sum = shm[shmOffset+threadId];
		float initial_sum = warp_sum;
		
		#pragma unroll 
		for(int i=1; i<warp_size; i*=2)
		{
			
			float n = shfl<float>::shfl_Up(warp_sum, i, warp_size);
			if (lane_id>= i)
				warp_sum+= n;
			
		}	
		
		shm[shmOffset+threadId]= warp_sum-initial_sum;
	}			
	__syncthreads();

	reg.x+=shm[shmOffset+warp_id];
	reg.y+=shm[shmOffset+warp_id];
	reg.z+=shm[shmOffset+warp_id];
	reg.w+=shm[shmOffset+warp_id];



	// Store the final result in global memory	
	dst[glbRPos]=reg;
}









// --- BranchTable -------------------------------------------------------


//- Template instantiation and branchtable for 'float' kernels
const static kernelCfg triXf32A[] = { //! GPU dependent
	NULL_ROW(1),
	NULL_ROW(2),
	ROW(KScanKS2,    4, 128,  4),
	ROW(KScanKS2,    8, 128,  4),
	ROW(KScanKS2,   16, 256,  4),
	ROW(KScanKS2,   32, 512,  4),
	ROW(KScanKS2,   64, 2048,  4),
	ROW(KScanKS2,  128, 1024,  4),
	ROW(KScanKS2,  256, 1024,  4),
	ROW(KScanKS2,  512, 1024,  4),
	ROW(KScanKS2, 1024,1024,  4),
	ROW(KScanKS2, 2048,2048,  4),
	ROW(KScanKS2, 4096,4096,  4),
	NULL_ROW(8192),
};


//- Template instantiation and branchtable for 'float' kernels
const static kernelCfg triXf32B[] = { //! GPU dependent
	NULL_ROW(1),
	NULL_ROW(2),
	ROW(KScanKS2,    4, 128,  4),
	ROW(KScanKS2,    8, 128,  4),
	ROW(KScanKS2,   16, 256,  4),
	ROW(KScanKS2,   32, 512,  4),
	ROW(KScanKS2,   64, 2048,  4),
	ROW(KScanKS2,  128, 1024,  4),
	ROW(KScanKS2,  256, 1024,  4),
	ROW(KScanKS2,  512, 1024,  4),
	ROW(KScanKS2, 1024,1024,  4),
	ROW(KScanKS2, 2048,2048,  4),
	ROW(KScanKS2, 4096,4096,  4),
	NULL_ROW(8192),
};


// --- Launcher ----------------------------------------------------------


//---- Interface Functions -----------------------------------------------

//- Main library function for 'float' scan
int KScanKS(float* input, float* output,
	int dir, int N, int stride, int batch)
{
	if(N>4096)
		return -1;

	
	return KLauncher2(tabla6, sizeof(tabla6),
		input, output, dir, N, batch);
}


