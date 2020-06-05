//- =======================================================================
//!+ Auxiliar CUDA kernels v1.1
//- =======================================================================

#pragma once
#ifndef CUDACL_HXX
#define CUDACL_HXX

//---- Header Dependencies -----------------------------------------------
#include <stdio.h>
#include <cuda.h>
#include "inc/complex.hxx"

//---- Macro Definitions -------------------------------------------------
#define kernel __global__ void
#define subkernel inline __device__
#define tFloat2 float2
#define tFloat float
#define MAX(_a, _b) ((_a) > (_b) ? (_a) : (_b))
#define MIN(_a, _b) ((_a) < (_b) ? (_a) : (_b))

//---- Math Functions ----------------------------------------------------
inline int
Log2(const float& fval) {
	int c = *(const int *) &fval;
	return (c >> 23) - 127;
}

inline int
Log2(const int& val) {
	float fval = (float)val;
	int c = *(const int *) &fval;
	return (c >> 23) - 127;
}

//---- Kernel Synchronization --------------------------------------------

enum mem_fence_flags { // Tipos de barrera
	LOCAL_MEM_FENCE  = 1, // OpenCL: Afecta a la memoria compartida
	GLOBAL_MEM_FENCE = 2, // OpenCL: Afecta a la memoria global
	DEVICE_MEM_FENCE = 4, // CUDA  : Afecta al dispositivo entero
	SYSTEM_MEM_FENCE = 8  // CUDA  : Afecta a todo el sistema
};

// Barrera de ejecucion
subkernel void
sync(mem_fence_flags flags) {
	// Barrera de dispositivo a nivel de sistema (Cuda Cap 2.x)
	// if(flags & SYSTEM_MEM_FENCE) {
	//	__threadfence_system(); return;
	// }
	// Barrera de dispositivo a nivel de kernel
	if(flags & DEVICE_MEM_FENCE) {
		__threadfence(); return;
	}
	// Barrera de memoria global dentro del bloque
	if(flags & GLOBAL_MEM_FENCE) {
		__threadfence_block(); return;
	}
	// Barrera de memoria compartida dentro del bloque
	if(flags & LOCAL_MEM_FENCE) {
		__syncthreads(); return;
	}
}

//---- ThreadId and GroupId ----------------------------------------------

subkernel int
get_global_id(const int dim) {
	switch(dim) {
		case 0 : return threadIdx.x + blockIdx.x * blockDim.x;
		case 1 : return threadIdx.y + blockIdx.y * blockDim.y;
		case 2 : return threadIdx.z + blockIdx.z * blockDim.z;
		default: return 0;
	}
}

subkernel int
get_local_id(const int dim) {
	switch(dim) {
		case 0 : return threadIdx.x;
		case 1 : return threadIdx.y;
		case 2 : return threadIdx.z;
		default: return 0;
	}
}

subkernel int
get_group_id(const int dim) {
	switch(dim) {
		case 0 : return blockIdx.x;
		case 1 : return blockIdx.y;
		case 2 : return blockIdx.z;
		default: return 0;
	}
}

subkernel int
get_local_size(const int dim) {
	switch(dim) {
		case 0 : return blockDim.x;
		case 1 : return blockDim.y;
		case 2 : return blockDim.z;
		default: return 1;
	}
}

subkernel int
get_num_groups(const int dim) {
	switch(dim) {
		case 0 : return gridDim.x;
		case 1 : return gridDim.y;
		case 2 : return gridDim.z;
		default: return 1;
	}
}

subkernel int
get_global_size(const int dim) {
	switch(dim) {
		case 0 : return blockDim.x * gridDim.x;
		case 1 : return blockDim.y * gridDim.y;
		case 2 : return blockDim.z * gridDim.z;
		default: return 1;
	}
}

subkernel int
get_work_dim() {
	return 1 + (gridDim.y > 1) + (gridDim.z > 1);
}

#endif // CUDACL_HXX

