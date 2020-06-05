//- =======================================================================
//+ Legacy RealFT kernels v1.0
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "tools/cudacl.hxx"
#include "inc/complex.hxx"
#include "inc/op_copy.hxx"
#include "inc/op_scale.hxx"
#include "inc/op_radix.hxx"
#include "KLauncher.hxx"

//---- Real radix kernels ------------------------------------------------

template<int N, int RAD, int DIR, int SHM> subkernel void
realRadix(COMPLEX* data, int threadXY) {
	COMPLEX reg[2];
	sync(LOCAL_MEM_FENCE); //- SYNC before copy
	for(int iter = threadXY; iter < SHM / 2; iter += SHM / RAD) {
		int base = 2*iter & ~(N-1);
		int posi = iter & (N/2-1);
		int posj = posi ? N-posi : N/2;
		float ang = 3.14159265f / N * posi;
		copy<2>(reg, data + base + posi, posj - posi);
		rbutterfly<DIR>(reg[0], reg[1], ang);
		copy<2>(data + base + posi, posj - posi, reg);
	}
}


//------- Cuda Kernels ---------------------------------------------------

//- RealFT basada en savestate 3
template<int N, int DIR, int RAD, int SHM> kernel
LegaRT(COMPLEX* src, int size) {
	// Identificadores de grupo-1D, thread-X y batch-Y
	int groupId = get_group_id(0) + get_group_id(1) * get_num_groups(0);
	int threadId = get_local_id(0);      // Thread horizontal (N)
	int batchId = get_local_id(1);       // Thread vertical (batch)
	int threadXY = threadId + batchId * get_local_size(0);

	// Desplazamientos para acceso a datos
	int shmPos = batchId * N + threadId; // Posicion actual en ShMem
	int srcPos = groupId * SHM + shmPos; // Posicion actual en GlbMem
	int dstPos = groupId * SHM + threadXY; // Posicion actual en GlbMem

	// Reservamos registros y memoria compartida
	COMPLEX reg[RAD];
	__shared__ COMPLEX shm[SHM];

	if(DIR >= 0) // Cargamos los datos de GlMem a registros	para etapa 1
		copy<RAD>(reg, src + srcPos, N / RAD);

	if(DIR < 0) { // (else) pre-procesado en la inverse RealFT
		// Cargamos los datos de GlMem a ShMem de forma coalescente
		copy<RAD>(shm + threadXY, SHM / RAD, src + dstPos, SHM / RAD);
		// Procesamos los datos en ShMem con la funcion RealRadix
		realRadix<N, RAD, DIR, SHM>(shm, threadXY);
		// Cargamos los datos de ShMem a registros para etapa 1 (Mix-Rad)
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(reg, shm + shmPos, N / RAD);
	}

	// Ejecutamos la primera etapa mixed-radix con datos en registros
	const int MIXRAD = MIXR<N, RAD>::val;
	if(DIR < 0) scale<RAD, N>(reg);
	radix<RAD, MIXRAD, DIR>(reg, RAD/MIXRAD); // stride = RAD/MIXRAD
	
	// Procesamor el resto de las etapas
	int cont = LOG2<N>::val - LOG2<MIXRAD>::val; // recomendable tener accRad y cont
	#pragma unroll
	for(int accRad = MIXRAD; accRad < N; accRad *= RAD) {

		// Etapa de reordenamiento: escritura
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(shm + shmPos, N / RAD, reg);

		cont -= LOG2<RAD>::val;
		int stride = 1 << cont;
		int readPos = batchId * N +
			(threadId & (stride-1)) | ((threadId & ~(stride-1)) * RAD);

		// Etapa de reordenamiento: lectura
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(reg, shm + readPos, stride);
		
		// Etapa de computacion
		float ang = getAngle<DIR, RAD>(accRad, threadId >> cont);
		radix<RAD, DIR>(reg, ang);
	}

	if(DIR >= 0) { // Post-procesado en la forward RealFT
		// Almacenamos los datos de registros en ShMem para filtrarlos
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(shm + shmPos, N / RAD, reg);
		// Procesamos los datos en ShMem con la funcion RealRadix
		realRadix<N, RAD, DIR, SHM>(shm, threadXY);
		// Guardamos los datos de ShMem a GlMem de forma coalescente
		sync(LOCAL_MEM_FENCE); //- SYNC before copy
		copy<RAD>(src + dstPos, SHM / RAD, shm + threadXY, SHM / RAD);
	}

	if(DIR < 0) // (else) Guardamos de registros a memoria global
		copy<RAD>(src + srcPos, N / RAD, reg);
}


// --- BranchTable -------------------------------------------------------

// BranchTable con templates instanciados y configuracion de lanzamiento
const static kernelCfg<COMPLEX*> rtTableX[] = { // GK110
	NULL_ROW(1),
	ROW(LegaRT,   2, 256, 2),
	ROW(LegaRT,   4, 256, 2),
	ROW(LegaRT,   8, 256, 2),
	ROW(LegaRT,  16, 256, 2),
	ROW(LegaRT,  32, 256, 2),
	ROW(LegaRT,  64, 512, 4),
	ROW(LegaRT, 128, 512, 4),
	ROW(LegaRT, 256, 512, 4),
	ROW(LegaRT, 512, 512, 4),
	ROW(LegaRT,1024,1024, 4),
	ROW(LegaRT,2048,2048, 8),
	ROW(LegaRT,4096,4096, 8),
	NULL_ROW(8192)
};

const static kernelCfg<COMPLEX*> rtTableY[] = { // GK110
	NULL_ROW(1),/*
	ROW(VertRT,   2, 128, 2),
	ROW(VertRT,   4, 256, 4),
	ROW(VertRT,   8, 512, 4),
	ROW(VertRT,  16, 512, 8),
	ROW(VertRT,  32, 512, 8),
	ROW(VertRT,  64, 512, 8),
	ROW(VertRT, 128, 512, 8),
	ROW(VertRT, 256,1024, 8),
	ROW(VertRT, 512,2048, 8),
	ROW(VertRT,1024,4096, 8),*/
	NULL_ROW(2048)
};


//---- Interface Functions -----------------------------------------------

//- Main library function: Gets the right configuration from the kernel tables
int KReal(float2* data, int dir, int N, int M, int batch) {
	int errVal = 0;
	if(N * M < 2) return -1; // Minimo 4 elementos
		
	if(N > 1) { // -- Transformada horizontal -----------------------
		const int TableSize = sizeof(rtTableX) / sizeof(rtTableX[0]);
		const int log2N = Log2(N);
		if(log2N >= TableSize) return -1;
		errVal += KLauncher(rtTableX[log2N], data, dir, N, 1, M * batch);
	}

	if(M > 1) { // -- Transformada vertical -------------------------
		const int TableSize = sizeof(rtTableY) / sizeof(rtTableY[0]);
		const int log2M = Log2(M);
		if(log2M >= TableSize) return -1;
		errVal += KLauncher(rtTableY[log2M], data, dir, N, M, batch);
	}
	
	return errVal;
}


//---- Older launcher -----------------------------------------------------

/*
const int R = 2;
const int D = 1;
const int S = 256;

int
KReal(float2* data, int dir, int N, int batch) {
	// BranchTable de instanciacion de templates
	const int TableSize = 13, TableOff = 1;

	static void(*fftTable[TableSize][2])(COMPLEX*, int) = {
		{ LegaRT<   2, D, R, MAX(   2, S)>, LegaRT<    2, -D, R, MAX(   2, S)> },
		{ LegaRT<   4, D, R, MAX(   4, S)>, LegaRT<    4, -D, R, MAX(   4, S)> },
		{ LegaRT<   8, D, R, MAX(   8, S)>, LegaRT<    8, -D, R, MAX(   8, S)> },
		{ LegaRT<  16, D, R, MAX(  16, S)>, LegaRT<   16, -D, R, MAX(  16, S)> },
		{ LegaRT<  32, D, R, MAX(  32, S)>, LegaRT<   32, -D, R, MAX(  32, S)> },
		{ LegaRT<  64, D, R, MAX(  64, S)>, LegaRT<   64, -D, R, MAX(  64, S)> },
		{ LegaRT< 128, D, R, MAX( 128, S)>, LegaRT<  128, -D, R, MAX( 128, S)> },
		{ LegaRT< 256, D, R, MAX( 256, S)>, LegaRT<  256, -D, R, MAX( 256, S)> },
		{ LegaRT< 512, D, R, MAX( 512, S)>, LegaRT<  512, -D, R, MAX( 512, S)> },
		{ LegaRT<1024, D, R, MAX(1024, S)>, LegaRT< 1024, -D, R, MAX(1024, S)> },
		#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ > 130)
		{ LegaRT<2048, D, R, MAX(2048, S)>, LegaRT< 2048, -D, R, MAX(2048, S)> },
		{ LegaRT<4096, D, R, MAX(4096, S)>, LegaRT< 4096, -D, R, MAX(4096, S)> },
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
	if(!blocks) puts("* FIXME: Increase batch size *");
	int blocksX = blocks > 32768 ? 32768 : blocks;
	int blocksY = blocks > 32768 ? blocks / 32768 : 1;

	// Lanzamos el kernel y comprobamos el codigo de error
	dim3 threadsPerBlock(threadsX, threadsY);
	dim3 blocksPerGrid(blocksX, blocksY);
	cudaFuncSetCacheConfig((const char*)kerPtr, cudaFuncCachePreferShared);
	kerPtr<<<blocksPerGrid, threadsPerBlock>>>(data, N * batch);
	return 0;
}
*/


//---- Temporal code -----------------------------------------------------

/*

// Empaqueta los datos iterando de forma coalescente
// dir = 1 -> buffer[i] = ...
kernel packR(float2* buffer, float* rdata, int halfN, int dir) {
	int posX = get_global_id(0);
	int stride = get_global_size(0) * get_global_size(1);

	// Version forward
	for(int i = posX; i < halfN * (dir > 0); i+= stride) {
		buffer[i].x = rdata[2*i  ];
		buffer[i].y = rdata[2*i+1];
	}

	// Version inversa
	for(int i = posX; i < halfN * (dir < 0); i+= stride) {
		rdata[2*i  ] *= buffer[i].x;
		rdata[2*i+1] *= buffer[i].y;
	}
}

// Empaqueta los datos iterando de forma coalescente
// dir = 1 -> buffer[i] = ...
kernel packC(float2* buffer, float2* cdata, int halfN, int dir) {
	int posX = get_global_id(0);
	int stride = get_global_size(0) * get_global_size(1);

	// Version forward
	for(int i = posX; i < halfN * (dir > 0); i+= stride) {
		buffer[i] = cdata[i];
		if(i == 0) buffer[0].y = cdata[halfN].x;
	}

	// Version inversa
	for(int i = posX; i < halfN * (dir < 0); i+= stride) {
		cdata[i      ] = buffer[i];
		if(i == 0) buffer[0].y = 0;
		cdata[i+halfN] = conj(buffer[(i > 0)*(halfN-i)]);
		if(i == 0) buffer[halfN] = make_COMPLEX(buffer[halfN].y, 0);
	}
}

// Threads que colaboran en el escalado
const int global_threads = 256;

void
pack(float2* buffer, float2* cdata, int N, int batch, int dir) {
	// Configuramos el lanzamiento del kernel
	int halfN = N >> 1;
	int blocks = (halfN + global_threads - 1) / global_threads;
	int blockX = blocks > 32768 ? 32768 : blocks;
	int blockY = blocks > 32768 ? blocks / 32768 : 1;
	dim3 threadsPerBlock(global_threads, 1);
	dim3 blocksPerGrid(blockX, blockY);
	cudaFuncSetCacheConfig((const char*)packC, cudaFuncCachePreferL1);

	// Lanzamos el kernel y comprobamos el codigo de error
	packC<<<blocksPerGrid, threadsPerBlock>>>(buffer, cdata, halfN, dir);
	if(cudaThreadSynchronize()) exit(printf("Error: pack\n"));
}

void
pack(float2* buffer, float* rdata, int N, int batch, int dir) {
	// Configuramos el lanzamiento del kernel
	int halfN = N >> 1;
	int blocks = (halfN + global_threads - 1) / global_threads;
	int blockX = blocks > 32768 ? 32768 : blocks;
	int blockY = blocks > 32768 ? blocks / 32768 : 1;
	dim3 threadsPerBlock(global_threads, 1);
	dim3 blocksPerGrid(blockX, blockY);
	cudaFuncSetCacheConfig((const char*)packR, cudaFuncCachePreferL1);

	// Lanzamos el kernel y comprobamos el codigo de error
	packR<<<blocksPerGrid, threadsPerBlock>>>(buffer, rdata, halfN, dir);
	if(cudaThreadSynchronize()) exit(printf("Error: pack\n"));
}

void LegacyRealFT::updateC(int dir) {
	int N = swap->getX() >> 1;
	int batch = swap->getY() * swap->getZ();
	Complex* data = gpuVector_c()->gpuData();
	pack((float2*)swap->gpuData(), (float2*)data, N, batch, dir);
	if(dir == -1) updateCpu_c();
}

void LegacyRealFT::updateR(int dir) {
	int N = swap->getX() >> 1;
	int batch = swap->getY() * swap->getZ();
	float* data = gpuVector_r()->gpuData();
	pack((float2*)swap->gpuData(), (float2*)data, N, batch, dir);
	if(dir == -1) updateCpu_r();
}
*/

/*
const static kernelCfg fftTableX[] = { // GF110
	{ NULL                    , NULL                    , 0,   0 },
	{ LegaRT<   2, D, 2, 256>, LegaRT<    2, -D, 2, 256>, 2, 256 },
	{ LegaRT<   4, D, 2, 256>, LegaRT<    4, -D, 2, 256>, 2, 256 },
	{ LegaRT<   8, D, 2, 256>, LegaRT<    8, -D, 2, 256>, 2, 256 },
	{ LegaRT<  16, D, 4, 512>, LegaRT<   16, -D, 4, 512>, 4, 512 },
	{ LegaRT<  32, D, 4, 512>, LegaRT<   32, -D, 4, 512>, 4, 512 },
	{ LegaRT<  64, D, 4, 512>, LegaRT<   64, -D, 4, 512>, 4, 512 },
	{ LegaRT< 128, D, 4, 512>, LegaRT<  128, -D, 4, 512>, 4, 512 },
	{ LegaRT< 256, D, 4, 512>, LegaRT<  256, -D, 4, 512>, 4, 512 },
	{ LegaRT< 512, D, 8, 512>, LegaRT<  512, -D, 8, 512>, 8, 512 },
	{ LegaRT<1024, D, 8,1024>, LegaRT< 1024, -D, 8,1024>, 8,1024 },
	#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ > 130)
	{ LegaRT<2048, D,16,2048>, LegaRT< 2048, -D,16,2048>,16,2048 },
	{ LegaRT<4096, D, 8,4096>, LegaRT< 4096, -D, 8,4096>, 8,4096 },
	#endif
	{ NULL                    , NULL                    , 0,   0 }
};
*/



