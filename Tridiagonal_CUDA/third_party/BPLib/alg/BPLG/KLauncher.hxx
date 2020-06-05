//- =======================================================================
//+ Kernel launcher v1.1
//- =======================================================================

#pragma once
#ifndef _KLAUNCHER
#define _KLAUNCHER

//---- Header Dependencies -----------------------------------------------

template<typename T> 
	struct kernelCfg {               // Estructura para los kernels
	void(*kerPtr[2])(T, int);       // Kernels FWD / INV
	short R;                         // Radix deseado
	short S;                         // Memoria compartida
}; 

// Inicializa una configuracion nula
#define NULL_ROW(_N) { NULL, NULL, 0, 0 }

// Inicializa una configuracion con los parametros especificados
#define ROW(_Kernel, _N, _S, _R) { \
	_Kernel<_N, 1, _R, _S>, \
    _Kernel<_N,-1, _R, _S>, \
	_R, _S }


//---- Function Declaration ----------------------------------------------

//- Kernel launcher: Configures threads and blocks before launch
template<typename T> int
KLauncher(const kernelCfg<T*> &launchCfg, T* dataPtr,
	int dir, int N, int M, int batch) {

	// Obtenemos el puntero al kernel corresponiente
	void(*kerPtr)(T*, int) = launchCfg.kerPtr[dir < 0];
	if(!kerPtr) return -1;

	// Comprobamos el tamaño de la entrada
	const int vSize = N * M * batch;
	if(launchCfg.S > vSize)
		return printf("* FIXME: Input size < %i *\n", launchCfg.S), -1;

	// Configuramos los threads para el lanzamiento del kernel
	int threadsX, threadsY, threadsZ = 1;
	if(M == 1) {                       //! -- Transformada horizontal --
		threadsX = N / launchCfg.R;    // Threads colaborando
		threadsY = launchCfg.S / N;    // Threads en batch
	} else {                           //! -- Transformada vertical --
		threadsX = MIN(N, 4);          // Threads para coalescencia
		threadsY = M / launchCfg.R;    // Threads colaborando
		threadsZ = launchCfg.S / (M * threadsX); // Threads en batch
	}
	dim3 threadsPerBlock(threadsX, threadsY, threadsZ); // Geometria threads

	// Comprobamos que la configuracion de threads es valida
	int numThreads = threadsX * threadsY * threadsZ;
	if(numThreads < 1 || numThreads > 1024 || threadsZ > 64) return -1;

	// Configuramos los bloques para el lanzamiento del kernel
	int numBlocks = vSize / launchCfg.S; // Bloques totales
	if(numBlocks * launchCfg.S != vSize)
		return printf("* FIXME: Incomplete block, size %i\n", launchCfg.S), -1;

	// Debe ser un tamaño valido para CUDA, por lo que se descompone en dos factores
	int blocksX = numBlocks, blocksY = 1;
	for( ; blocksX > 32768 && blocksY < blocksX; blocksY <<=1, blocksX >>= 1)
		if(blocksX & 0x01) break;     // No se puede seguir factorizando en 2^x
	dim3 blocksPerGrid(blocksX, blocksY); // Geometria final del bloque

	// Lanzamos el kernel y comprobamos el codigo de error
	cudaFuncSetCacheConfig((const char*)kerPtr, cudaFuncCachePreferShared);
	kerPtr<<<blocksPerGrid, threadsPerBlock>>>(dataPtr, N);

	return 0;
}

#endif // _KLAUNCHER
