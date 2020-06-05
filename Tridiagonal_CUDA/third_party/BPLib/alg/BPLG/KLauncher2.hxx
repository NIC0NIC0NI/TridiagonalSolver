//- =======================================================================
//+ Kernel launcher v1.2
//- =======================================================================

#pragma once
#ifndef _KLAUNCHER2
#define _KLAUNCHER2

//---- Header Dependencies -----------------------------------------------

//- Declare kernel prototype for BranchTables
struct kernelCfg { // Kernels structure
	void(*kerPtr)(const float4*, float4*, int);
	short R;                         // Radix 
	short S;                         // Shared memory
};

//- Initializes a null configuration for unsupported sizes
#define NULL_ROW(_N) { NULL, 0, 0 }

//- Initializes a configuration entry with the specified parameters
#define ROW(_Kernel, _N, _S, _R) { \
	_Kernel< _N, 1, _R, (_N > _S ? _N : _S)>, \
	_R, (_N > _S ? _N : _S) }




//---- Function Declaration ----------------------------------------------

//- Kernel launcher: Configures threads and blocks before launch
template<typename T> int
KLauncher2(const kernelCfg* cfgTable, int tableBytes,
	T* srcPtr, T* dstPtr, int dir, int N, int batch) {


	
	// Check configuration and data pointers are valid
	if(!cfgTable || !srcPtr || !dstPtr || !batch)
		return printf("* FIXME: Bad launcher parameters *\n"), -1;

	

	// Verify that the selected N is in the table range
	const int TableSize = tableBytes / sizeof(kernelCfg);
	const int log2N = Log2(N);
	if(log2N < 0 || log2N >= TableSize) return -1;	
	

	// Obtains the corresponding kernel configuration
	const kernelCfg launchCfg = cfgTable[log2N];
	if(!launchCfg.kerPtr) return -1;
	

	// Check for valid input size
	const int vSize = N * batch;   //! For tridiagonal, dim M=4
	if((launchCfg.S - 1) & vSize)  // previously: launchCfg.S > vSize
		return printf("* FIXME: Input size not divisible by %i *\n", launchCfg.S), -1;

	// Configure thread geometry for kernel launch
	int threadsX, threadsY, threadsZ = 1;
	threadsX = N / launchCfg.R;    // Threads colaborando
	threadsY = launchCfg.S / N;    // Threads en batch
	dim3 threadsPerBlock(threadsX, threadsY, threadsZ);

	// Valite thread geometry configuration
	const int MaxThreads = 1024;   // Max threads per block
	int numThreads = threadsX * threadsY * threadsZ;
	if(numThreads < 1 || numThreads > MaxThreads || threadsZ > 64) return -1;

	// Obtain required total blocks
	int numBlocks = vSize / launchCfg.S;
	if(numBlocks * launchCfg.S != vSize)
		return printf("* FIXME: Incomplete block, size%%%i\n", launchCfg.S), -1;

	// Block geometry decomposition (for cards with CUDA caps 2.0)
	int blocksX = numBlocks, blocksY = 1;
	for( ; blocksX > 32768 && blocksY < blocksX; blocksY <<=1, blocksX >>= 1)
		if(blocksX & 0x01) break;         // Can't factorize further by 2^x
	dim3 blocksPerGrid(blocksX, blocksY); // Final block geometry

	// Set cache configuration and launch kernel
	cudaFuncSetCacheConfig((const char*)launchCfg.kerPtr, cudaFuncCachePreferShared);
	// cudaFuncSetSharedMemConfig((const char*)kerPtr, cudaSharedMemBankSizeEightByte);
	launchCfg.kerPtr<<<blocksPerGrid, threadsPerBlock>>>((float4*)srcPtr, (float4*) dstPtr, N);
	
	// Cuda kernel calls are asynchronous, sync to check execution errors
	return 0;
}

#endif // _KLAUNCHER
