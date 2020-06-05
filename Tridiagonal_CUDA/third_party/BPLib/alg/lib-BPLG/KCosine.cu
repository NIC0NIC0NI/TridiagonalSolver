//- =======================================================================
//+ BPLG Cosine
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "tools/cudacl.hxx"
#include "inc/complex.hxx"
#include "inc/op_copy.hxx"
#include "inc/op_scale.hxx"
#include "inc/op_radix.hxx"
#include "KLauncher.hxx"

//---- Helper kernels ------------------------------------------------

template<int N> subkernel void
copyDCT(float* dstData, int dstStride, const COMPLEX* srcData, int srcStride = 1) {
	#pragma unroll
    for(int i = 0; i < N; i++)
        dstData[i * dstStride] = srcData[i * srcStride].x;
}

template<int N> subkernel void
copyDCT(COMPLEX* dstData, int dstStride, const float* srcData, int srcStride = 1) {
	#pragma unroll
    for(int i = 0; i < N; i++)
        dstData[i * dstStride] = make_COMPLEX(srcData[i * srcStride], 0.0f);
}

template<int N, int RAD, int DIR, int SHM> subkernel void
radixDCT(COMPLEX* data, int threadId) {
	const int STRIDE = N / RAD;
	float fac = DIR * -3.14159265f / 2.0f / N;
	#pragma unroll
	for(int iter = 0; iter < RAD; iter++) {
		COMPLEX ang;
		int i = threadId + iter * STRIDE;
		fastCosSin(i * fac, ang);
		data[iter] = ang * data[iter];
	}
}

template<int N, int RAD, int DIR, int SHM> subkernel void
packDCT(COMPLEX* dst, COMPLEX* src, int threadId, int batchId) {
	const int STRIDE = N / RAD;
	int base = batchId * N;
	#pragma unroll
	for(int iter = 0; iter < RAD; iter++) {
		int i = threadId + iter * STRIDE;
		int j = i < N/2 ? 2*i : 2*(N-i)-1;
		if(DIR > 0) dst[iter] = src[base + j];
		if(DIR < 0) dst[base + j] = src[iter];
	}
}


//------- Cuda Kernels ---------------------------------------------------

template<int N, int DIR, int RAD, int SHM> kernel
LegaDCT(float* src, int size) {
	// Identificadores de grupo-1D, thread-X y batch-Y
	int groupId = get_group_id(0) + get_group_id(1) * get_num_groups(0);
	int threadId = get_local_id(0);      // Thread horizontal (N)
	int batchId = get_local_id(1);       // Thread vertical (batch)
	int threadXY = threadId + batchId * get_local_size(0);

	// Desplazamientos para acceso a datos
	int glbStride = N / RAD;
	int shmOffset = batchId * N;
	int shmPos = threadId + shmOffset;     // Posicion actual en ShMem
	int glbPos = groupId * SHM + shmPos;   // Posicion actual en GlbMem
	int dctPos = groupId * SHM + threadXY; // Posicion aplanada en GlbMem

	// Reservamos registros y memoria compartida
	COMPLEX reg[RAD];
	__shared__ COMPLEX shm[SHM];

	// Cargamos los datos *reales* desde memoria global
	if(DIR > 0) {
		copyDCT<RAD>(shm + threadXY, SHM / RAD, src + dctPos, SHM / RAD);
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		packDCT<N, RAD, DIR, SHM>(reg, shm, threadId, batchId);
	}
	if(DIR < 0) {
		copyDCT<RAD>(reg, 1, src + glbPos, glbStride);
		radixDCT<N, RAD, DIR, SHM>(reg, threadId);
		if(threadId == 0) reg[0].x *= 0.5f;
	}

	// Ejecutamos siempre la primera etapa mixed-radix
	const int MIXRAD = MIXR<N, RAD>::val;
	if(DIR < 0) scale<RAD, N/2>(reg);
	radix<RAD, MIXRAD, DIR>(reg, RAD/MIXRAD); // stride = RAD/MIXRAD

	// Procesamor el resto de las etapas
	int cont = LOG2<N>::val - LOG2<MIXRAD>::val; // recomendable tener accRad y cont
	for(int accRad = MIXRAD; accRad < N; accRad *= RAD) {

		// Calculamos el offset y el stride
		cont -= LOG2<RAD>::val;
		int stride = 1 << cont;
		int readPos = shmOffset +
			(threadId & (stride-1)) | ((threadId & ~(stride-1)) * RAD);

		// Etapa de reordenamiento: escritura
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(shm + shmPos, N / RAD, reg);

		// Etapa de reordenamiento: lectura
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(reg, shm + readPos, stride);
		
		// Etapa de computacion
		float ang = getAngle<DIR, RAD>(accRad, threadId >> cont);
		radix<RAD, DIR>(reg, ang);
	}

	// Guardamos el resultado en memoria compartida
	sync(LOCAL_MEM_FENCE); //- SYNC before copy
	if(DIR > 0) {
		radixDCT<N, RAD, DIR, SHM>(reg, threadId);
		copy<RAD>(shm + shmPos, N / RAD, reg);
	}
	if(DIR < 0) {
		packDCT<N, RAD, DIR, SHM>(shm, reg, threadId, batchId);
	}

	// Guardamos los datos *reales* a memoria global
	sync(LOCAL_MEM_FENCE); //- SYNC before copy
	copyDCT<RAD>(src + dctPos, SHM / RAD, shm + threadXY, SHM / RAD);
}


template<int N, int DIR, int RAD, int SHM> kernel
VertDCT(float* src, int stride) {
	// Identificadores de grupo-1D, thread-X y batch-Y
	int coalesceId = get_local_id(0);  // Threads para coalescencia
	int coalesceSize = get_local_size(0); // Total threads para coalescencia
	int threadId = get_local_id(1);    // Threads que colaboran en N
	int batchId = coalesceId + get_local_id(2) * coalesceSize;  // Batch en grupo
	int groupIdG = get_group_id(0) + get_group_id(1) * get_num_groups(0);
	int groupIdGB = get_local_id(2) + get_local_size(2) * groupIdG;
	int groupId = coalesceId + coalesceSize * groupIdGB;

	// Desplazamientos para acceso a datos
	int offsetX = groupId & (stride - 1); // Desplazamiento sobre 'stride'
	int offsetY = groupId / stride; // Desplazamiento sobre batch de problemas
	int glbStride = stride * N / RAD;
	int shmOffset = N * batchId;
	int shmPos = threadId + shmOffset;
	int glbPos = offsetX + stride * (threadId + offsetY * N);

	// Reservamos registros y memoria compartida
	COMPLEX reg[RAD];
	__shared__ COMPLEX shm[SHM];

	// Cargamos los datos *reales* desde memoria global
	if(DIR > 0) {
		copyDCT<RAD>(shm + shmPos, N / RAD, src + glbPos, glbStride);
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		packDCT<N, RAD, DIR, SHM>(reg, shm, threadId, batchId);
	}
	if(DIR < 0) {
		copyDCT<RAD>(reg, 1, src + glbPos, glbStride);
		radixDCT<N, RAD, DIR, SHM>(reg, threadId);
		if(threadId == 0) reg[0].x *= 0.5f;
	}

	// Ejecutamos siempre la primera etapa mixed-radix
	const int MIXRAD = MIXR<N, RAD>::val;
	if(DIR < 0) scale<RAD, N/2>(reg);
	radix<RAD, MIXRAD, DIR>(reg, RAD/MIXRAD); // stride = RAD/MIXRAD

	// Procesamor el resto de las etapas
	int cont = LOG2<N>::val - LOG2<MIXRAD>::val; // recomendable tener accRad y cont
	for(int accRad = MIXRAD; accRad < N; accRad *= RAD) {

		// Calculamos el offset y el stride
		cont -= LOG2<RAD>::val;
		int stride = 1 << cont;
		int readPos = shmOffset +
			(threadId & (stride-1)) | ((threadId & ~(stride-1)) * RAD);

		// Etapa de reordenamiento: escritura
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(shm + shmPos, N / RAD, reg);

		// Etapa de reordenamiento: lectura
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(reg, shm + readPos, stride);
		
		// Etapa de computacion
		float ang = getAngle<DIR, RAD>(accRad, threadId >> cont);
		radix<RAD, DIR>(reg, ang);
	}

	// Guardamos el resultado en memoria compartida
	sync(LOCAL_MEM_FENCE); //- SYNC before copy
	if(DIR > 0) {
		radixDCT<N, RAD, DIR, SHM>(reg, threadId);
		copy<RAD>(shm + shmPos, N / RAD, reg);
	}
	if(DIR < 0) {
		packDCT<N, RAD, DIR, SHM>(shm, reg, threadId, batchId);
	}

	// Guardamos los datos *reales* a memoria global
	sync(LOCAL_MEM_FENCE); //- SYNC before copy
	copyDCT<RAD>(src + glbPos, glbStride, shm + shmPos, N / RAD);
}


// --- BranchTable -------------------------------------------------------

// BranchTable con templates instanciados y configuracion de lanzamiento
const static kernelCfg<float*> dctTableX[] = { // GK110
	NULL_ROW(1),
	ROW(LegaDCT,   2, 256, 2),
	ROW(LegaDCT,   4, 256, 2),
	ROW(LegaDCT,   8, 256, 2),
	ROW(LegaDCT,  16, 256, 2),
	ROW(LegaDCT,  32, 256, 2),
	ROW(LegaDCT,  64, 512, 4),
	ROW(LegaDCT, 128, 512, 4),
	ROW(LegaDCT, 256, 512, 4),
	ROW(LegaDCT, 512, 512, 4),
	ROW(LegaDCT,1024,1024, 4),
	ROW(LegaDCT,2048,2048, 8),
	ROW(LegaDCT,4096,4096, 8),
	NULL_ROW(8192)
};

const static kernelCfg<float*> dctTableY[] = { // GK110
	NULL_ROW(1),
	ROW(VertDCT,   2, 128, 2),
	ROW(VertDCT,   4, 256, 4),
	ROW(VertDCT,   8, 512, 4),
	ROW(VertDCT,  16, 512, 8),
	ROW(VertDCT,  32, 512, 8),
	ROW(VertDCT,  64, 512, 8),
	ROW(VertDCT, 128, 512, 8),
	ROW(VertDCT, 256,1024, 8),
	ROW(VertDCT, 512,2048, 8),
	ROW(VertDCT,1024,4096, 8),
	NULL_ROW(2048)
};


//---- Interface Functions -----------------------------------------------

//- Main library function: Gets the right configuration from the kernel tables
int KCosine(float* data, int dir, int N, int M, int batch) {
	int errVal = 0;
		
	if(N > 1) { // -- Transformada horizontal -----------------------
		const int TableSize = sizeof(dctTableX) / sizeof(dctTableX[0]);
		const int log2N = Log2(N);
		if(log2N >= TableSize) return -1;
		errVal += KLauncher(dctTableX[log2N], data, dir, N, 1, M * batch);
	}

	if(M > 1) { // -- Transformada vertical -------------------------
		const int TableSize = sizeof(dctTableY) / sizeof(dctTableY[0]);
		const int log2M = Log2(M);
		if(log2M >= TableSize) return -1;
		errVal += KLauncher(dctTableY[log2M], data, dir, N, M, batch);
	}
	
	return errVal;
}









// 	//if(threadId == 0) shm[shmPos] = reg[0]; // Posicion 0 para normalizar

/*
template<int N> subkernel void
printfv(const COMPLEX* data, const char* label = "") {
	printf("%s:\n", label);
	#pragma unroll
	for(int i = 0; i < N; i++)
		printf("%i> %7.2f,%+7.2fi\n", i, data[i].x, data[i].y);
}
*/


