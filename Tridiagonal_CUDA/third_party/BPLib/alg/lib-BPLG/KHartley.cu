//- =======================================================================
//+ Legacy Hartley kernels v1.0
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "tools/cudacl.hxx"
#include "inc/complex.hxx"
#include "inc/op_copy.hxx"
#include "inc/op_scale.hxx"
#include "inc/op_radix.hxx"
#include "KLauncher.hxx"

//---- Hartley helper kernels --------------------------------------------

template<int N> subkernel void
copyHT(float* dstData, int dstStride, const COMPLEX* srcData) {
	#pragma unroll
    for(int i = 0; i < N; i++)
        dstData[i * dstStride] = srcData[i].x - srcData[i].y;
}

template<int N> subkernel void
copyHT(COMPLEX* dstData, const float* srcData, int srcStride) {
	#pragma unroll
    for(int i = 0; i < N; i++)
        dstData[i] = make_COMPLEX(srcData[i * srcStride], 0.0f);
}

//------- Cuda Kernels ---------------------------------------------------

template<int N, int DIR, int RAD, int SHM> kernel
LegaHT(float* src, int size) {
	// Identificadores de grupo-1D, thread-X y batch-Y
	int groupId = get_group_id(0) + get_group_id(1) * get_num_groups(0);
	int threadId = get_local_id(0);      // Thread horizontal (N)
	int batchId = get_local_id(1);       // Thread vertical (batch)

	// Desplazamientos para acceso a datos
	int glbStride = N / RAD;
	int shmOffset = batchId * N;
	int shmPos = threadId + shmOffset;     // Posicion actual en ShMem
	int glbPos = groupId * SHM + shmPos; // Posicion actual en GlbMem

	// Reservamos registros y memoria compartida
	COMPLEX reg[RAD];
	__shared__ COMPLEX shm[N > RAD ? SHM : 1];

	// Cargamos los datos *reales* desde memoria global
	copyHT<RAD>(reg, src + glbPos, glbStride);

	// Ejecutamos siempre la primera etapa mixed-radix
	const int MIXRAD = MIXR<N, RAD>::val;
	if(DIR < 0) scale<RAD, N>(reg);
	radix<RAD, MIXRAD, 1>(reg, RAD/MIXRAD); // stride = RAD/MIXRAD
	
	// Procesamor el resto de las etapas
	int cont = LOG2<N>::val - LOG2<MIXRAD>::val; // recomendable tener accRad y cont
	for(int accRad = MIXRAD; accRad < N; accRad *= RAD) {

		// Calculamos el offset y el stride
		cont -= LOG2<RAD>::val;
		int stride = 1 << cont;
		int readPos = shmOffset +
			(threadId & (stride-1)) | ((threadId & ~(stride-1)) * RAD);

		// Etapa de reordenamiento: escritura
		if(accRad != MIXRAD)       //- After the first iteration...
			sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(shm + shmPos, N / RAD, reg);

		// Etapa de reordenamiento: lectura
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(reg, shm + readPos, stride);
		
		// Etapa de computacion (mul DIR = 1)
		float ang = getAngle<1, RAD>(accRad, threadId >> cont);
		radix<RAD, 1>(reg, ang);
	}

	// Guardamos los datos a memoria global
	copyHT<RAD>(src + glbPos, glbStride, reg);
}


//- Transformada vertical, multiplexion para coalescencia sin incrementar ShMem
template<int N, int DIR, int RAD, int SHM> kernel
VertHT(float* src, int stride) {
	// Identificadores de grupo-1D, thread-X y batch-Y
	int coalesceId = get_local_id(0);  // Threads para coalescencia
	int coalesceSize = get_local_size(0); // Total threads para coalescencia
	int threadId = get_local_id(1);    // Threads que colaboran en N
	int batchId = coalesceId + get_local_id(2) * coalesceSize; // Batch en grupo
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
	__shared__ COMPLEX shm[N > RAD ? SHM : 1];

	// Cargamos los datos desde memoria global
	copyHT<RAD>(reg, src + glbPos, glbStride);

	// Ejecutamos siempre la primera etapa mixed-radix
	const int MIXRAD = MIXR<N, RAD>::val;
	if(DIR < 0) scale<RAD, N>(reg);
	radix<RAD, MIXRAD, 1>(reg, RAD/MIXRAD); // stride = RAD/MIXRAD
	
	// Procesamor el resto de las etapas
	int cont = LOG2<N>::val - LOG2<MIXRAD>::val; // recomendable tener accRad y cont
	for(int accRad = MIXRAD; accRad < N; accRad *= RAD) {

		// Calculamos el offset y el stride
		cont -= LOG2<RAD>::val;
		int stride = 1 << cont;
		int readPos = shmOffset +
			(threadId & (stride-1)) | ((threadId & ~(stride-1)) * RAD);

		// Etapa de reordenamiento: escritura
		if(accRad != MIXRAD)       //- After the first iteration...
			sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(shm + shmPos, N / RAD, reg);

		// Etapa de reordenamiento: lectura
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(reg, shm + readPos, stride);

		// Etapa de computacion
		float ang = getAngle< 1, RAD>(accRad, threadId >> cont);
		radix<RAD, 1>(reg, ang);
	}

	// Guardamos los datos a memoria global
	copyHT<RAD>(src + glbPos, glbStride, reg);
}


// --- BranchTable -------------------------------------------------------

// BranchTable con templates instanciados y configuracion de lanzamiento
const static kernelCfg<float*> htTableX[] = { // GK110
	NULL_ROW(1),
	ROW(LegaHT,   2, 256, 2),
	ROW(LegaHT,   4, 256, 2),
	ROW(LegaHT,   8, 256, 2),
	ROW(LegaHT,  16, 256, 2),
	ROW(LegaHT,  32, 256, 2),
	ROW(LegaHT,  64, 512, 4),
	ROW(LegaHT, 128, 512, 4),
	ROW(LegaHT, 256, 512, 4),
	ROW(LegaHT, 512, 512, 4),
	ROW(LegaHT,1024,1024, 4),
	ROW(LegaHT,2048,2048, 8),
	ROW(LegaHT,4096,4096, 8),
	NULL_ROW(8192)
};

const static kernelCfg<float*> htTableY[] = { // GK110
	NULL_ROW(1),
	ROW(VertHT,   2, 128, 2),
	ROW(VertHT,   4, 256, 4),
	ROW(VertHT,   8, 512, 4),
	ROW(VertHT,  16, 512, 8),
	ROW(VertHT,  32, 512, 8),
	ROW(VertHT,  64, 512, 8),
	ROW(VertHT, 128, 512, 8),
	ROW(VertHT, 256,1024, 8),
	ROW(VertHT, 512,2048, 8),
	ROW(VertHT,1024,4096, 8),
	NULL_ROW(2048)
};


//---- Interface Functions -----------------------------------------------

//- Main library function: Gets the right configuration from the kernel tables
int KHartley(float* data, int dir, int N, int M, int batch) {
	int errVal = 0;
		
	if(N > 1) { // -- Transformada horizontal -----------------------
		const int TableSize = sizeof(htTableX) / sizeof(htTableX[0]);
		const int log2N = Log2(N);
		if(log2N >= TableSize) return -1;
		errVal += KLauncher(htTableX[log2N], data, dir, N, 1, M * batch);
	}

	if(M > 1) { // -- Transformada vertical -------------------------
		const int TableSize = sizeof(htTableY) / sizeof(htTableY[0]);
		const int log2M = Log2(M);
		if(log2M >= TableSize) return -1;
		errVal += KLauncher(htTableY[log2M], data, dir, N, M, batch);
	}
	
	return errVal;
}
