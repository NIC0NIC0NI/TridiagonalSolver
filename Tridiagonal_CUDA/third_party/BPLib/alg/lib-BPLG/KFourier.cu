//- =======================================================================
//+ Legacy FFT kernels v2.1
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "tools/cudacl.hxx"
#include "inc/op_copy.hxx"
#include "inc/op_scale.hxx"
#include "inc/op_radix.hxx"
#include "KLauncher.hxx"

//---- Cuda Kernels ------------------------------------------------------
//- Savestate 3: Mas rapido que la version coalescente para N grande
//- La etapa de reordenamiento puede ser expresada simplemente como:
//- 'shmExSync<RAD>(reg, shm + shmPos, N / RAD, shm + readPos, stride);'
template<int N, int DIR, int RAD, int SHM> kernel
LegaFT(COMPLEX* src, int size) {
	// Identificadores de grupo-1D, thread-X y batch-Y
	int groupId = get_group_id(0) + get_group_id(1) * get_num_groups(0);
	int threadId = get_local_id(0);      // Thread horizontal (N)
	int batchId = get_local_id(1);       // Thread vertical (batch)

	// Desplazamientos para acceso a datos
	int glbStride = N / RAD;
	int shmOffset = batchId * N;
	int shmPos = threadId + shmOffset; // Posicion actual en ShMem
	int glbPos = groupId * SHM + shmPos; // Posicion actual en GlbMem

	// Reservamos registros y memoria compartida
	COMPLEX reg[RAD];
	__shared__ COMPLEX shm[N > RAD ? SHM : 1];

	// Cargamos los datos desde memoria global
	copy<RAD>(reg, src + glbPos, glbStride);
	
	// Ejecutamos siempre la primera etapa mixed-radix
	const int MIXRAD = MIXR<N, RAD>::val;
	if(DIR < 0) scale<RAD, N>(reg);
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
		if(accRad != MIXRAD)       //- After the first iteration...
			sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(shm + shmPos, N / RAD, reg);

		// Etapa de reordenamiento: lectura
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(reg, shm + readPos, stride);

		// Etapa de computacion
		float ang = getAngle<DIR, RAD>(accRad, threadId >> cont);
		radix<RAD, DIR>(reg, ang);
	}

	// Guardamos los datos a memoria global
	copy<RAD>(src + glbPos, glbStride, reg);
}


//- Transformada vertical, multiplexion para coalescencia sin incrementar ShMem
template<int N, int DIR, int RAD, int SHM> kernel
VertFT(COMPLEX* src, int stride) {
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
	copy<RAD>(reg, src + glbPos, glbStride);

	// Ejecutamos siempre la primera etapa mixed-radix
	const int MIXRAD = MIXR<N, RAD>::val;
	if(DIR < 0) scale<RAD, N>(reg);
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
		if(accRad != MIXRAD)       //- After the first iteration...
			sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(shm + shmPos, N / RAD, reg);

		// Etapa de reordenamiento: lectura
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(reg, shm + readPos, stride);

		// Etapa de computacion
		float ang = getAngle<DIR, RAD>(accRad, threadId >> cont);
		radix<RAD, DIR>(reg, ang);
	}

	// Guardamos los datos a memoria global
	copy<RAD>(src + glbPos, glbStride, reg);
}


// --- BranchTable -------------------------------------------------------

// BranchTable con templates instanciados y configuracion de lanzamiento
const static kernelCfg<COMPLEX*> fftTableX[] = { // GK110
	NULL_ROW(1),
	ROW(LegaFT,   2, 256, 2),
	ROW(LegaFT,   4, 256, 2),
	ROW(LegaFT,   8, 256, 2),
	ROW(LegaFT,  16, 256, 2),
	ROW(LegaFT,  32, 256, 2),
	ROW(LegaFT,  64, 512, 4),
	ROW(LegaFT, 128, 512, 4),
	ROW(LegaFT, 256, 512, 4),
	ROW(LegaFT, 512, 512, 4),
	ROW(LegaFT,1024,1024, 4),
	ROW(LegaFT,2048,2048, 8),
	ROW(LegaFT,4096,4096, 8),
	NULL_ROW(8192)
};

const static kernelCfg<COMPLEX*> fftTableY[] = { // GK110
	NULL_ROW(1),
	ROW(VertFT,   2, 128, 2),
	ROW(VertFT,   4, 256, 4),
	ROW(VertFT,   8, 512, 4),
	ROW(VertFT,  16, 512, 8),
	ROW(VertFT,  32, 512, 8),
	ROW(VertFT,  64, 512, 8),
	ROW(VertFT, 128, 512, 8),
	ROW(VertFT, 256,1024, 8),
	ROW(VertFT, 512,2048, 8),
	ROW(VertFT,1024,4096, 8),
	NULL_ROW(2048)
};

//! TODO: Hacer una version que use threadX colaboran X
//! threadY colaboran Y y batchZ en el mismo bloque
//! optimizada para problemas con NxM <= 64x64

//---- Interface Functions -----------------------------------------------

//- Main library function: Gets the right configuration from the kernel tables
int KFourier(float2* data, int dir, int N, int M, int batch) {
	int errVal = 0;
	
	if(N > 1) { // -- Transformada horizontal -----------------------
		const int TableSize = sizeof(fftTableX) / sizeof(fftTableX[0]);
		const int log2N = Log2(N);
		if(log2N >= TableSize) return -1;
		errVal += KLauncher(fftTableX[log2N], data, dir, N, 1, M * batch);
	}

	if(M > 1) { // -- Transformada vertical -------------------------
		const int TableSize = sizeof(fftTableY) / sizeof(fftTableY[0]);
		const int log2M = Log2(M);
		if(log2M >= TableSize) return -1;
		errVal += KLauncher(fftTableY[log2M], data, dir, N, M, batch);
	}
	
	return errVal;
}



/*
//? Experimento de multiplexion de la memoria compartida
//- Transformada vertical, multiplexion para coalescencia sin incrementar ShMem
template<int N, int DIR, int RAD, int SHM> kernel
VertFT(COMPLEX* src, int stride) {
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
	int shmOffset = N * (batchId >> 1);
	int shmPos = threadId + shmOffset;
	int glbPos = offsetX + stride * (threadId + offsetY * N);
	
	// Reservamos registros y memoria compartida
	COMPLEX reg[RAD];
	__shared__ COMPLEX shm[N < SHM ? SHM / 2 : SHM];

	// Cargamos los datos desde memoria global
	copy<RAD>(reg, src + glbPos, glbStride);

	// Ejecutamos siempre la primera etapa mixed-radix
	const int MIXRAD = MIXR<N, RAD>::val;
	if(DIR < 0) scale<RAD, N>(reg);
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
		if(batchId & 0x01 == 0)
			copy<RAD>(shm + shmPos, N / RAD, reg);
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		if(batchId & 0x01 == 1)
			copy<RAD>(shm + shmPos, N / RAD, reg);

		// Etapa de reordenamiento: lectura
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		if(batchId & 0x01 == 0)
			copy<RAD>(reg, shm + readPos, stride);
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		if(batchId & 0x01 == 1)
			copy<RAD>(reg, shm + readPos, stride);

		// Etapa de computacion
		float ang = getAngle<DIR, RAD>(accRad, threadId >> cont);
		radix<RAD, DIR>(reg, ang);
	}

	// Guardamos los datos a memoria global
	copy<RAD>(src + glbPos, glbStride, reg);
}
*/

/*
	//? Este lanzador permite estimar el rendimiento para problemas grandes
	if(N > 1024) {
		const int TableSize = sizeof(fftTableY) / sizeof(fftTableY[0]);
		const int a = 1024;
		const int b = N / 1024;
		const int log2a = Log2(a);
		const int log2b = Log2(b);
		if(log2M >= TableSize) return -1;
		errVal += KLauncher(fftTableY[log2a], data, dir, a, b, batch);
		errVal += KLauncher(fftTableY[log2b], data, dir, b, a, batch);
		return errVal;
	}

//- Transformada vertical, multiplexion para coalescencia sin incrementar ShMem
template<int N, int DIR, int RAD, int SHM> kernel
LongFT(const COMPLEX* __restrict__ src, COMPLEX* __restrict__ dst, int stride) {
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
	copy<RAD>(reg, src + glbPos, glbStride);

	// Ejecutamos siempre la primera etapa mixed-radix
	const int MIXRAD = MIXR<N, RAD>::val;
	if(DIR < 0) scale<RAD, N>(reg);
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
		if(accRad != MIXRAD)       //- After the first iteration...
			sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(shm + shmPos, N / RAD, reg);

		// Etapa de reordenamiento: lectura
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(reg, shm + readPos, stride);

		// Etapa de computacion
		float ang = getAngle<DIR, RAD>(accRad, threadId >> cont);
		radix<RAD, DIR>(reg, ang);
	}

	
	//! Aplicamos el twiddle para la segunda parte
	//? Hint: usar get_group_id(0) para obtener el angulo
	//int totalId = threadId + offsetX * N;
	float ang = getAngle<DIR, M>(N, offsetX);
	twiddle<RAD>(reg, ang); // Sabemos que es RAD en lugar de MIXRAD
	// 0 0 0 0, 0 1 2 3, 0 2 4 6, 0 3 6 9
	//copy<RAD>(shm + shmPos, N / RAD, reg);
	//copy<RAD>(reg, src + shmPos, N / RAD, reg);
	

	//! Escribimos los datos en horizontal (out of place)
	/// copy<RAD>(dst + threadId...

	// Guardamos los datos a memoria global
	// copy<RAD>(src + glbPos, glbStride, reg);
}
*/

/*

if(N == M && N * M <= 4096) { // -- Transformada doble -----------------
	const int TableSize = sizeof(fftTableXY) / sizeof(kernelCfg);
	const int log2N = Log2(N);
	if(log2N >= TableSize) return -1;
	return KLauncher(fftTableXY[log2N], data, dir, N, 1, M * batch);
}

const static kernelCfg fftTableXY[] = { // GK110
	{ NULL                   , NULL                   , 0,    0 },
	{ NULL,                  , NULL                   , 2,  256 },
	{ RectFT<  4, 1, 2,  256>, RectFT<  4,-1, 2,  256>, 2,  256 },
	{ RectFT<  8, 1, 2,  256>, RectFT<  8,-1, 2,  256>, 2,  256 },
	{ RectFT< 16, 1, 2,  256>, RectFT< 16,-1, 2,  256>, 2,  256 },
	{ RectFT< 32, 1, 4, 1024>, RectFT< 32,-1, 4, 1024>, 4, 1024 },
	{ RectFT< 64, 1, 8, 4096>, RectFT< 64,-1, 8, 4096>, 8, 4096 },
	{ NULL                   , NULL                   , 0,    0 }
};


template<int N, int DIR, int RAD, int SHM> kernel
RectFT(COMPLEX* src, int size) {
	// Identificadores de grupo-1D, thread-X y batch-Y
	int groupId = get_group_id(0) + get_group_id(1) * get_num_groups(0);
	int threadId = get_local_id(0);      // Thread horizontal (N)
	int batchId = get_local_id(1);       // Thread vertical (batch)

	// Desplazamientos para acceso a datos
	int glbStride = N / RAD;
	int shmOffset = batchId * N;
	int shmPos = threadId + shmOffset; // Posicion actual en ShMem
	int glbPos = groupId * SHM + shmPos; // Posicion actual en GlbMem

	// Reservamos registros y memoria compartida
	COMPLEX reg[RAD];
	__shared__ COMPLEX shm[N > RAD ? SHM : 1];

	// Cargamos los datos desde memoria global
	copy<RAD>(reg, src + glbPos, glbStride);
	
	// Ejecutamos siempre la primera etapa mixed-radix
	const int MIXRAD = MIXR<N, RAD>::val;
	if(DIR < 0) scale<RAD, N * N>(reg);
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
		if(accRad != MIXRAD)       //- After the first iteration...
			sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(shm + shmPos, N / RAD, reg);

		// Etapa de reordenamiento: lectura
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(reg, shm + readPos, stride);

		// Etapa de computacion
		float ang = getAngle<DIR, RAD>(accRad, threadId >> cont);
		radix<RAD, DIR>(reg, ang);
	}

	// Transponemos los datos
	if(N > RAD) sync(LOCAL_MEM_FENCE); //- SYNC before copy
	copy<RAD>(shm + shmPos, N / RAD, reg);
	
	sync(LOCAL_MEM_FENCE); //- SYNC before copy
	int offsetX = batchId & (N - 1);
	int offsetY = batchId >> LOG2<N>::val;
	int shmPosY = offsetX + threadId * N + offsetY * N * N;
	copy<RAD>(reg, shm + shmPosY, N * N / RAD);
	
	radix<RAD, MIXRAD, DIR>(reg, RAD/MIXRAD); // stride = RAD/MIXRAD
	
	// Procesamor el resto de las etapas
	cont = LOG2<N>::val - LOG2<MIXRAD>::val; // recomendable tener accRad y cont
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
	
	// Guardamos los datos a memoria global
	sync(LOCAL_MEM_FENCE); //- SYNC before copy
	copy<RAD>(shm + shmPosY, N * N / RAD, reg);
	
	sync(LOCAL_MEM_FENCE); //- SYNC before copy
	copy<RAD>(src + glbPos, glbStride, shm + shmPos, N / RAD);
}

*/



/*
const int R = 4;
const int D = 1;
const int S = 512;

int
KFourier(float2* data, int dir, int N, int batch) {
	// BranchTable de instanciacion de templates
	const int TableSize = 12, TableOff = 2;
	static void(*fftTable[TableSize][2])(COMPLEX*, int) = {
		{ LegaFT<   4, D, R, MAX(   4, S)>, LegaFT<    4, -D, R, MAX(   4, S)> },
		{ LegaFT<   8, D, R, MAX(   8, S)>, LegaFT<    8, -D, R, MAX(   8, S)> },
		{ LegaFT<  16, D, R, MAX(  16, S)>, LegaFT<   16, -D, R, MAX(  16, S)> },
		{ LegaFT<  32, D, R, MAX(  32, S)>, LegaFT<   32, -D, R, MAX(  32, S)> },
		{ LegaFT<  64, D, R, MAX(  64, S)>, LegaFT<   64, -D, R, MAX(  64, S)> },
		{ LegaFT< 128, D, R, MAX( 128, S)>, LegaFT<  128, -D, R, MAX( 128, S)> },
		{ LegaFT< 256, D, R, MAX( 256, S)>, LegaFT<  256, -D, R, MAX( 256, S)> },
		{ LegaFT< 512, D, R, MAX( 512, S)>, LegaFT<  512, -D, R, MAX( 512, S)> },
		{ LegaFT<1024, D, R, MAX(1024, S)>, LegaFT< 1024, -D, R, MAX(1024, S)> },
		#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ > 130)
		{ LegaFT<2048, D, R, MAX(2048, S)>, LegaFT< 2048, -D, R, MAX(2048, S)> },
		{ LegaFT<4096, D, R, MAX(4096, S)>, LegaFT< 4096, -D, R, MAX(4096, S)> },
		#endif
		{ NULL                    , NULL}
	};

	// Obtenemos el puntero al kernel corresponiente
	int log2N = Log2(N);
	if(log2N < TableOff || log2N >= TableSize + TableOff) return -1;
	void(*kerPtr)(COMPLEX*, int) = fftTable[log2N - TableOff][dir < 0];
	if(!kerPtr) return -1;

	// Configuramos el lanzamiento del kernel
	int threadsX = N / R;
	if(threadsX < 1 || threadsX > 1024) return -1;
	int threadsY = S > N ? S / N : 1;

	int blocks = batch / threadsY;
	int blocksX = blocks > 32768 ? 32768 : blocks;
	int blocksY = blocks > 32768 ? blocks / 32768 : 1;

	// Lanzamos el kernel y comprobamos el codigo de error
	dim3 threadsPerBlock(threadsX, threadsY);
	dim3 blocksPerGrid(blocksX, blocksY);
	cudaFuncSetCacheConfig((const char*)kerPtr, cudaFuncCachePreferShared);
	// cudaFuncSetSharedMemConfig((const char*)kerPtr, cudaSharedMemBankSizeEightByte );
	kerPtr<<<blocksPerGrid, threadsPerBlock>>>(data, N * batch);
	return 0;
}
*/



/*
// --- BranchTable -------------------------------------------------------

const int D = 1; // Asignar direccion D = 0 para modo solo reordenamiento

typedef struct {                     // Estructura para los kernels
	void(*kerPtr[2])(COMPLEX*, int); // Kernels FWD / INV
	short thX; short thY;            // Configuracion threads
	short thZ; short blX;            // Configuracion bloques
} kernelCfg;

// BranchTable con templates instanciados y configuracion de lanzamiento
const static kernelCfg fftTable[] = {
	{ NULL                    , NULL                  ,   0  ,      0, 0 },
	{ NULL                    , NULL                  ,   0  ,      0, 0 },
	{ LegaFT<   4, D, R, 256>, LegaFT<   4,-D, R, 256>,   4/R, S/   4, 1 },
	{ LegaFT<   8, D, R, 256>, LegaFT<   8,-D, R, 256>,   8/R, S/   8, 1 },
	{ LegaFT<  16, D, R, 256>, LegaFT<  16,-D, R, 256>,  16/R, S/  16, 1 },
	{ LegaFT<  32, D, R, 256>, LegaFT<  32,-D, R, 256>,  32/R, S/  32, 1 },
	{ LegaFT<  64, D, R, 256>, LegaFT<  64,-D, R, 256>,  64/R, S/  64, 1 },
	{ LegaFT< 128, D, R, 256>, LegaFT< 128,-D, R, 256>, 128/R, S/ 128, 1 },
	{ LegaFT< 256, D, R, 256>, LegaFT< 256,-D, R, 256>, 256/R, S/ 256, 1 },
	{ LegaFT< 512, D, R, 512>, LegaFT< 512,-D, R, 512>, 512/R, S/ 512, 1 },
	{ LegaFT<1024, D, R,1024>, LegaFT<1024,-D, R,1024>,1024/R, S/1024, 1 },
	#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ > 130)
	{ LegaFT<2048, D, R,2048>, LegaFT<2048,-D, R,2048>,2048/R, S/2048, 1 },
	{ LegaFT<4096, D, R,4096>, LegaFT<4096,-D, R,4096>,4096/R, S/4096, 1 },
	#endif
	{ NULL                    , NULL                  , 0    ,      0, 0 }
};
*/


//---- Temporal code -----------------------------------------------------


/*
//- Savestate 0: Version inicial del algoritmo
template<int N, int DIR, int RAD, int SHM> kernel
LegaFT(COMPLEX* src, int size) {
	// Identificadores de grupo-1D, thread-X y batch-Y
	int groupId = get_group_id(0) + get_group_id(1) * get_num_groups(0);
	int threadId = get_local_id(0);      // Thread horizontal (N)
	int batchId = get_local_id(1);       // Thread vertical (batch)
	int threadXY = threadId + batchId * get_local_size(0);

	// Desplazamientos para acceso a datos
	int shmPos = batchId * N + threadId; // Posicion actual en ShMem
	int glbPos = groupId * SHM + threadXY; // Posicion actual en GlbMem

	// Reservamos registros y memoria compartida
	COMPLEX reg[RAD];
	__shared__ COMPLEX shm[SHM];

	// Ejecutamos siempre la primera etapa mixed-radix
	const int MIXRAD = MIXR<N, RAD>::val;
	sync(LOCAL_MEM_FENCE); //+ SYNC before copy
	copy<RAD>(reg, src + shmPos, N / RAD);
	if(DIR < 0) scale<RAD, N>(reg);
	radix<RAD, MIXRAD, DIR>(reg, RAD/MIXRAD); // stride = RAD/MIXRAD
	sync(LOCAL_MEM_FENCE); //+ SYNC before copy
	copy<RAD>(shm + shmPos, N / RAD, reg);

	// Procesamor el resto de las etapas
	//? #pragma unroll
	for(int accRad = MIXRAD; accRad < N; accRad *= RAD) {
		int stride = N / RAD / accRad;
		int readPos = batchId * N +
			(threadId & (stride-1)) | ((threadId & ~(stride-1)) * RAD);
		sync(LOCAL_MEM_FENCE); //+ SYNC before copy
		copy<RAD>(reg, shm + readPos, stride);
		float ang = getAngle<DIR, RAD>(accRad, threadId); //? runtime accRad

		radix<RAD, DIR>(reg, ang);
		sync(LOCAL_MEM_FENCE);//+ SYNC before copy
		copy<RAD>(shm + shmPos, N / RAD, reg);
	}

	// Guardamos los datos a memoria global
	sync(LOCAL_MEM_FENCE); //+ SYNC before copy
	copy<RAD>(src + glbPos, SHM / RAD, shm  + threadXY, SHM / RAD);
}
*/


/*
//- Savestate 1: Acceso coalescente a memoria
template<int N, int DIR, int RAD, int SHM> kernel
LegaFT(COMPLEX* src, int size) {
	// Identificadores de grupo-1D, thread-X y batch-Y
	int groupId = get_group_id(0) + get_group_id(1) * get_num_groups(0);
	int threadId = get_local_id(0);      // Thread horizontal (N)
	int batchId = get_local_id(1);       // Thread vertical (batch)
	int threadXY = threadId + batchId * get_local_size(0);

	// Desplazamientos para acceso a datos
	int shmPos = batchId * N + threadId; // Posicion actual en ShMem
	int glbPos = groupId * SHM + threadXY; // Posicion actual en GlbMem

	// Reservamos registros y memoria compartida
	COMPLEX reg[RAD];
	__shared__ COMPLEX shm[SHM];


	// Cargamos los datos desde memoria global
	// copy<RAD>(shm + threadXY, SHM / RAD, src + glbPos, SHM / RAD);
	// sync(LOCAL_MEM_FENCE); //+ SYNC before copy
	// copy<RAD>(reg, shm + shmPos, N / RAD);

	// Ejecutamos siempre la primera etapa mixed-radix
	const int MIXRAD = MIXR<N, RAD>::val;
	copy<RAD>(reg, src + groupId * SHM + shmPos, N / RAD);
	if(DIR < 0) scale<RAD, N>(reg);
	radix<RAD, MIXRAD, DIR>(reg, RAD/MIXRAD); // stride = RAD/MIXRAD
	sync(LOCAL_MEM_FENCE); //+ SYNC before copy
	copy<RAD>(shm + shmPos, N / RAD, reg);

	// Procesamor el resto de las etapas
	int cont = LOG2<N>::val - LOG2<MIXRAD>::val; // recomendable tener accRad y cont
	//? #pragma unroll
	for(int accRad = MIXRAD; accRad < N; accRad *= RAD) {
		cont -= LOG2<RAD>::val;
		int stride = 1 << cont;
		int readPos = batchId * N +
			(threadId & (stride-1)) | ((threadId & ~(stride-1)) * RAD);
		sync(LOCAL_MEM_FENCE); //+ SYNC before copy
		copy<RAD>(reg, shm + readPos, stride);
		
		float ang = -6.2831853f * (float)DIR / RAD / accRad * (threadId >> cont);
		radix<RAD, DIR>(reg, ang);

		sync(LOCAL_MEM_FENCE);//+ SYNC before copy
		copy<RAD>(shm + shmPos, N / RAD, reg);		
	}

	// Guardamos los datos a memoria global
	sync(LOCAL_MEM_FENCE); //+ SYNC before copy
	copy<RAD>(src + glbPos, SHM / RAD, shm + threadXY, SHM / RAD);
	
}
*/

/*
//- Savestate 2: Solo para reutilizar los bucles printf para debug
template<int N, int DIR, int RAD, int SHM> kernel
LegaFT(COMPLEX* src, int size) {
	// Identificadores de grupo-1D, thread-X y batch-Y
	int groupId = get_group_id(0) + get_group_id(1) * get_num_groups(0);
	int threadId = get_local_id(0);      // Thread horizontal (N)
	int batchId = get_local_id(1);       // Thread vertical (batch)
	int threadXY = threadId + batchId * get_local_size(0);

	// Desplazamientos para acceso a datos
	int shmPos = batchId * N + threadId; // Posicion actual en ShMem
	int glbPos = groupId * SHM + threadXY; // Posicion actual en GlbMem

	// Reservamos registros y memoria compartida
	COMPLEX reg[RAD];
	__shared__ COMPLEX shm[SHM];

	//! debug, colorea la memoria compartid
	//	if(groupId + batchId + threadId == 0)
	//		for(int i = 0; i < SHM; i++) { shm[i].x = 123.45f; shm[i].y = 678.90f; }
	//	__syncthreads();

	// Cargamos los datos desde memoria global
	//? if(glbPos < size) //! Inhabilita threads extra
	//copy<RAD>(shm + threadXY, SHM / RAD, src + glbPos, SHM / RAD);

	//! debug, muestra informacion sobre la ShMem despues de cargar los datos
	// __syncthreads();
	// if(groupId == 0)
	//		printf("%i,%i: reading %i elements from %i..%i to %i..%i with stride %i\n",
	//		threadId, batchId, RAD, glbPos, glbPos + SHM-SHM/RAD,
	//		shmPos, shmPos + SHM-SHM/RAD, SHM/RAD);
	//	if(groupId + batchId + threadId == 0) {
	//		for(int i = 0; i < SHM; i+=4)
	//			printf("S1|%i> %7.2f%+7.2fi, %7.2f%+7.2fi, %7.2f%+7.2fi, %7.2f%+7.2fi\n",
	//			i, shm[i].x, shm[i].y, shm[i+1].x, shm[i+1].y,
	//			shm[i+2].x, shm[i+2].y, shm[i+3].x, shm[i+3].y);
	//	}

	// Ejecutamos siempre la primera etapa mixed-radix
	const int MIXRAD = MIXR<N, RAD>::val;
	//sync(LOCAL_MEM_FENCE); //+ SYNC before copy
	copy<RAD>(reg, src + groupId * SHM + shmPos, N / RAD);
	//copy<RAD>(reg, shm + shmPos, N / RAD);
	if(DIR < 0) scale<RAD, N>(reg);
	radix<RAD, MIXRAD, DIR>(reg, RAD/MIXRAD); // stride = RAD/MIXRAD
	sync(LOCAL_MEM_FENCE); //+ SYNC before copy
	copy<RAD>(shm + shmPos, N / RAD, reg);

	//! debug, muestra informacion sobre los registros en el primer radix
	//	__syncthreads();
	//	if(groupId == 0)
	//		printf("%i,%i: stride %i, computing Rad-%i and MixRad-%i\n",
	//		threadId, batchId, N/RAD, RAD, MIXRAD);
	//	
	//	if(groupId + batchId + threadId == 0) {
	//		for(int i = 0; i < RAD; i+=4)
	//			printf("R2|%i= %7.2f%+7.2fi, %7.2f%+7.2fi, %7.2f%+7.2fi, %7.2f%+7.2fi\n",
	//			i, reg[i].x, reg[i].y, reg[i+1].x, reg[i+1].y,
	//			reg[i+2].x, reg[i+2].y, reg[i+3].x, reg[i+3].y);
	//	}
	//	if(groupId + batchId + threadId == 0) {
	//		for(int i = 0; i < SHM; i+=4)
	//			printf("S2|%i> %7.2f%+7.2fi, %7.2f%+7.2fi, %7.2f%+7.2fi, %7.2f%+7.2fi\n",
	//			i, shm[i].x, shm[i].y, shm[i+1].x, shm[i+1].y,
	//			shm[i+2].x, shm[i+2].y, shm[i+3].x, shm[i+3].y);
	//	}
	
	// Procesamor el resto de las etapas
	//? #pragma unroll
	for(int accRad = MIXRAD; accRad < N; accRad *= RAD) {
		int stride = N / RAD / accRad;
		int readPos = batchId * N +
			(threadId & (stride-1)) | ((threadId & ~(stride-1)) * RAD);
		sync(LOCAL_MEM_FENCE); //+ SYNC before copy
		copy<RAD>(reg, shm + readPos, stride);
		float ang = getAngle<DIR, RAD>(accRad, threadId); //? runtime accRad
		//float ang = 3.1415926f / accRad * (float)DIR * (threadId & (accRad-1));

		//! debug, muestra informacion sobre los registros en cada etapa radix
		//	__syncthreads();
		//			if(groupId + batchId + threadId == 0) {
		//				for(int i = 0; i < RAD; i+=4)
		//					printf("R3|%i= %7.2f%+7.2fi, %7.2f%+7.2fi, %7.2f%+7.2fi, %7.2f%+7.2fi\n",
		//						i, reg[i].x, reg[i].y, reg[i+1].x, reg[i+1].y,
		//						reg[i+2].x, reg[i+2].y, reg[i+3].x, reg[i+3].y);
		//			}
		//			if(groupId + batchId == 0)
		//				printf("%i> radix stage: accrad = %i, stride = %i, readPos = %i, angle=%f\n",
		//				threadId, accRad, stride, readPos, ang);

		radix<RAD, DIR>(reg, ang);
		sync(LOCAL_MEM_FENCE);//+ SYNC before copy
		copy<RAD>(shm + shmPos, N / RAD, reg);
		//			/*
		//			if(groupId + batchId + threadId == 0) {
		//				for(int i = 0; i < RAD; i+=4)
		//					printf("R3|%i= %7.2f%+7.2fi, %7.2f%+7.2fi, %7.2f%+7.2fi, %7.2f%+7.2fi\n",
		//						i, reg[i].x, reg[i].y, reg[i+1].x, reg[i+1].y,
		//						reg[i+2].x, reg[i+2].y, reg[i+3].x, reg[i+3].y);
		//			}
	}

	//! debug, muestra informacion sobre la ShMem al finalizar los calculos

	//	__syncthreads();
	//	if(groupId + batchId + threadId == 0) {
	//		for(int i = 0; i < SHM; i+=4)
	//			printf("S3|%i< %7.2f%+7.2fi, %7.2f%+7.2fi, %7.2f%+7.2fi, %7.2f%+7.2fi\n",
	//			i, shm[i].x, shm[i].y, shm[i+1].x, shm[i+1].y,
	//			shm[i+2].x, shm[i+2].y, shm[i+3].x, shm[i+3].y);
	//	}
	
	// Guardamos los datos a memoria global
	//? if(glbPos < size) //! Inhabilita threads extra
	sync(LOCAL_MEM_FENCE); //+ SYNC before copy
	copy<RAD>(src + glbPos, SHM / RAD, shm  + threadXY, SHM / RAD);
	// copy<RAD>(src + glbPos, N / RAD, reg);
}
*/

/*

//- Savestate 3: Mas rapido que la version coalescente para N grande (SHM/2)
template<int N, int DIR, int RAD, int SHM> kernel
LegaFT(COMPLEX* src, int size) {
	// Identificadores de grupo-1D, thread-X y batch-Y
	int groupId = get_group_id(0) + get_group_id(1) * get_num_groups(0);
	int threadId = get_local_id(0);      // Thread horizontal (N)
	int batchId = get_local_id(1);       // Thread vertical (batch)

	// Desplazamientos para acceso a datos
	int shmPos = batchId * N + threadId; // Posicion actual en ShMem
	int glbPos = groupId * SHM + shmPos; // Posicion actual en GlbMem

	// Reservamos registros y memoria compartida
	COMPLEX reg[RAD];
	__shared__ float shm[SHM];
	
	// Cargamos los datos desde memoria global
	copy<RAD>(reg, src + glbPos, N / RAD);
	
	// Ejecutamos siempre la primera etapa mixed-radix
	const int MIXRAD = MIXR<N, RAD>::val;
	if(DIR < 0) scale<RAD, N>(reg);
	radix<RAD, MIXRAD, DIR>(reg, RAD/MIXRAD); // stride = RAD/MIXRAD
	
	// Procesamor el resto de las etapas
	int cont = LOG2<N>::val - LOG2<MIXRAD>::val; // recomendable tener accRad y cont
	for(int accRad = MIXRAD; accRad < N; accRad *= RAD) {

		cont -= LOG2<RAD>::val;
		int stride = 1 << cont;
		int readPos = batchId * N +
			(threadId & (stride-1)) | ((threadId & ~(stride-1)) * RAD);
		
		shmExSync<RAD, 1>(reg, 0, shm + shmPos, N / RAD, shm + readPos, stride);

		// Etapa de computacion
		float ang = getAngle<DIR, RAD>(accRad, threadId >> cont);
		radix<RAD, DIR>(reg, ang);
	}

	// Guardamos los datos a memoria global
	copy<RAD>(src + glbPos, N / RAD, reg);
}

*/


/*
Pasada Horizontal + pasada vertical


//---- Cuda Kernels ------------------------------------------------------
//- Savestate 3: Mas rapido que la version coalescente para N grande
template<int N, int M, int DIR, int RAD, int SHM> kernel
MiniFT(COMPLEX* src, int size) {
	const int MIXRADX = MIXR<N, RAD>::val;
	const int MIXRADY = MIXR<M, RAD>::val;

	// Identificadores de grupo-1D, thread-X y batch-Y
	int groupId = get_group_id(0) + get_group_id(1) * get_num_groups(0);
	int threadIdX = get_local_id(0);      // Thread horizontal (N)
	int threadIdY = get_local_id(1);       // Thread vertical (batch)

	// Desplazamientos para acceso a datos
	int shmPosX = threadIdY * N + threadIdX; // Posicion actual en ShMem
	int shmPosY = threadIdY + threadIdX * M; // Posicion actual en ShMem
	int glbPos = groupId * SHM + shmPos; // Posicion actual en GlbMem

	// Reservamos registros y memoria compartida
	COMPLEX reg[RAD];
	__shared__ COMPLEX shm[SHM];

	// Cargamos los datos desde memoria global
	copy<RAD>(reg, src + glbPos, N / RAD);
	if(DIR < 0) scale<RAD, N * M>(reg);
	
	// Ejecutamos siempre la primera etapa mixed-radix
	radix<RAD, MIXRADX, DIR>(reg, RAD/MIXRADX); // stride = RAD/MIXRADX
	
	// Procesamor el resto de las etapas
	int cont1 = LOG2<N>::val - LOG2<MIXRADX>::val; // recomendable tener accRad y cont
	for(int accRad = MIXRADX; accRad < N; accRad *= RAD) {

		// Etapa de reordenamiento: escritura
		if(accRad != MIXRADX)       //- After the first iteration...
			sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(shm + shmPosX, N / RAD, reg);

		cont1 -= LOG2<RAD>::val;
		int stride = 1 << cont1;
		int readPos = batchId * N +
			(threadIdX & (stride-1)) | ((threadIdX & ~(stride-1)) * RAD);

		// Etapa de reordenamiento: lectura
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(reg, shm + readPos, stride);

		// Etapa de computacion
		float ang = getAngle<DIR, RAD>(accRad, threadIdX >> cont);
		radix<RAD, DIR>(reg, ang);
	}

	// Guardamos los datos a memoria compartida (vertical)
	copy<RAD>(shm + shmPosX, N / RAD);

	// Cargamos los datos desde memoria compartida (horizontal)
	copy<RAD>(reg, shm + shmPosY, M / RAD);

	// Ejecutamos siempre la primera etapa mixed-radix
	radix<RAD, MIXRADY, DIR>(reg, RAD/MIXRADY); // stride = RAD/MIXRADX
	
	// Procesamor el resto de las etapas
	int cont2 = LOG2<M>::val - LOG2<MIXRADY>::val; // recomendable tener accRad y cont
	for(int accRad = MIXRADY; accRad < M; accRad *= RAD) {

		// Etapa de reordenamiento: escritura
		if(accRad != MIXRADY)       //- After the first iteration...
			sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(shm + shmPosY, M / RAD, reg);

		cont2 -= LOG2<RAD>::val;
		int stride = 1 << cont2;
		int readPos = threadIdX * M +
			(threadIdY & (stride-1)) | ((threadIdY & ~(stride-1)) * RAD);

		// Etapa de reordenamiento: lectura
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(reg, shm + readPos, stride);

		// Etapa de computacion
		float ang = getAngle<DIR, RAD>(accRad, threadIdY >> cont);
		radix<RAD, DIR>(reg, ang);
	}

	// Cargamos los datos desde memoria compartida (horizontal)
	copy<RAD>(shm + shmPosY, M / RAD, reg);

	// Guardamos los datos a memoria global
	copy<RAD>(src + glbPos, N / RAD, shm + shmPosX, N / RAD);
}

*/

/*

//- Codigo usado para procesar problemas 2D en un solo kernel
//- Actualmente no funciona por un bug en el diseño de threads

const static kernelCfg fftTableXY[] = { // GK110
	{ NULL                       , NULL                       , 0,    0 },
	{ RectFT<  2,  2, 0, 2,  256>, RectFT<  2,  2,-0, 2,  256>, 2,  128 },
	{ RectFT<  4,  4, 0, 4,  512>, RectFT<  4,  4,-0, 4,  512>, 4,  512 },
	{ RectFT<  8,  8, 0, 8, 2048>, RectFT<  8,  8,-0, 8, 2048>, 8, 2048 },
	{ RectFT< 16, 16, 0, 4,  256>, RectFT< 16, 16,-0, 4,  256>, 4,  256 },
	{ RectFT< 32, 32, 0, 8, 1024>, RectFT< 32, 32,-0, 8, 1024>, 8, 1024 },
	{ RectFT< 64, 64, 0, 8, 4096>, RectFT< 64, 64,-0, 8, 4096>, 8, 4096 },
	{ NULL                       , NULL                       , 0,    0 }
};

//---- Cuda Kernels ------------------------------------------------------
//- Savestate 3: Mas rapido que la version coalescente para N grande
template<int N, int M, int DIR, int RAD, int SHM> kernel
RectFT(COMPLEX* src, int size) {
	// Identificadores de grupo-1D, thread-X y batch-Y
	int groupId = get_group_id(0) + get_group_id(1) * get_num_groups(0);
	int threadIdX = get_local_id(0);      // Thread horizontal (N)
	int threadIdY = get_local_id(1);      // Thread vertical (M)
	int batchId = get_local_id(2);       // Thread en profundidad (batch)
	int batchIdX = threadIdY + batchId * get_local_size(1);
	int batchIdY = threadIdX + batchId * get_local_size(0);

	// Desplazamientos para acceso a datos
	int glbStride = N / RAD;
	int shmOffsetX = batchIdX * N;
	int shmPosX = threadIdX + shmOffsetX; // Posicion actual en ShMem
	int glbPos = groupId * SHM + shmPosX; // Posicion actual en GlbMem

	// Reservamos registros y memoria compartida
	COMPLEX reg[RAD];
	__shared__ COMPLEX shm[SHM];

	// Cargamos los datos desde memoria global
	copy<RAD>(reg, src + glbPos, glbStride);
	
	// Ejecutamos siempre la primera etapa mixed-radix
	const int MIXRADN = MIXR<N, RAD>::val;
	if(DIR < 0) scale<RAD, N * M>(reg);
	radix<RAD, MIXRADN, DIR>(reg, RAD/MIXRADN); // stride = RAD/MIXRAD
	
	// Procesamor el resto de las etapas
	int cont = LOG2<N>::val - LOG2<MIXRADN>::val; // recomendable tener accRad y cont
	for(int accRad = MIXRADN; accRad < N; accRad *= RAD) {

		// Calculamos el offset y el stride
		cont -= LOG2<RAD>::val;
		int stride = 1 << cont;
		int readPos = shmOffsetX +
			(threadIdX & (stride-1)) | ((threadIdX & ~(stride-1)) * RAD);

		// Etapa de reordenamiento: escritura
		if(accRad != MIXRADN)       //- After the first iteration...
			sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(shm + shmPosX, N / RAD, reg);

		// Etapa de reordenamiento: lectura
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(reg, shm + readPos, stride);

		// Etapa de computacion
		float ang = getAngle<DIR, RAD>(accRad, threadIdX >> cont);
		radix<RAD, DIR>(reg, ang);
	}

	// Guardamos los datos a memoria compartida
	if(N > RAD) sync(LOCAL_MEM_FENCE); //- SYNC before copy
	copy<RAD>(shm + shmPosX, N / RAD, reg);

	/// ----------------------------------------------------
	int offsetX = batchIdY & (N - 1); // Desplazamiento sobre 'stride'
	int offsetY = batchIdY / N; // Desplazamiento sobre batch de problemas
	int shmOffsetY = batchIdY * M;
	int shmPosY = threadIdY * N + offsetX + offsetY * N * M;
	
	// Cargamos los datos de memoria compartida
	sync(LOCAL_MEM_FENCE); //- SYNC before copy
	copy<RAD>(reg, shm + shmPosY, N * (M / RAD));
	
	if(threadIdX == 0 && threadIdY == 0 && batchId == 1 && groupId == 0)
	for(int i = 0; i < RAD; i++) {
		printf("1: %i= %7.2f%+7.2fi\n", i, reg[i].x, reg[i].y);
	}
	
	// Ejecutamos siempre la primera etapa mixed-radix
	const int MIXRADM = MIXR<M, RAD>::val;
	radix<RAD, MIXRADM, DIR>(reg, RAD/MIXRADM); // stride = RAD/MIXRAD
	
	if(threadIdX == 0 && threadIdY == 0 && batchId == 1 && groupId == 0)
	for(int i = 0; i < RAD; i++) {
		printf("2: %i= %7.2f%+7.2fi\n", i, reg[i].x, reg[i].y);
	}
	
	// Procesamor el resto de las etapas
	cont = LOG2<M>::val - LOG2<MIXRADM>::val; // recomendable tener accRad y cont
	for(int accRad = MIXRADM; accRad < M; accRad *= RAD) {

		// Calculamos el offset y el stride
		cont -= LOG2<RAD>::val;
		int stride = 1 << cont;
		int readPos = shmOffsetY +
			(threadIdY & (stride-1)) | ((threadIdY & ~(stride-1)) * RAD);

		// Etapa de reordenamiento: escritura
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(shm + threadIdY + shmOffsetY, M / RAD, reg);// <--

		// Etapa de reordenamiento: lectura
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(reg, shm + readPos, stride);

		// Etapa de computacion
		float ang = getAngle<DIR, RAD>(accRad, threadIdY >> cont);
		radix<RAD, DIR>(reg, ang);
	}
	
	// Guardamos los datos a memoria compartida
	sync(LOCAL_MEM_FENCE); //- SYNC before copy
	copy<RAD>(shm + shmPosY, N * (M / RAD), reg);
	
	// Guardamos los datos a memoria global
	sync(LOCAL_MEM_FENCE); //- SYNC before copy
	copy<RAD>(src + glbPos, glbStride, shm + shmPosX, N / RAD);
}



*/


/*

//- Subkernel para calcular la FFT
template<int N, int DIR, int RAD, int SHM> subkernel void
FFT_helper(COMPLEX* reg, COMPLEX* shm,
	const int threadId, const int shmPos, const int shmOffset)
{
	// Ejecutamos siempre la primera etapa mixed-radix
	const int MIXRAD = MIXR<N, RAD>::val;
	if(DIR < 0) scale<RAD, N>(reg);
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
}
*/

