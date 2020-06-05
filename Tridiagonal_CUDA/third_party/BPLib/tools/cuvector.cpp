//- =======================================================================
//+ Cuda Vector Class v3.0
//- =======================================================================

#pragma once
#ifndef CUVector_CPP
#define CUVector_CPP
#include "cuvector.hxx"

//---- Include Section ---------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

//---- Gestion de objetos ------------------------------------------------

// Constructor 3D (reserva memoria tanto en la CPU como en la GPU)
template <typename T>
CUVector<T>::CUVector(int dimX, int dimY, int dimZ, T* cpuBuf, T* gpuBuf) :
		_dimX(dimX), _dimY(dimY), _dimZ(dimZ),
		_cpuData(cpuBuf), _gpuData(gpuBuf) { objInit(); }

// Constructor 2D (reserva memoria tanto en la CPU como en la GPU)
template <typename T>
CUVector<T>::CUVector(int dimX, int dimY, T* cpuBuf, T* gpuBuf) :
		_dimX(dimX), _dimY(dimY), _dimZ(1),
		_cpuData(cpuBuf), _gpuData(gpuBuf) { objInit(); }

// Constructor 1D (reserva memoria tanto en la CPU como en la GPU)
template <typename T>
CUVector<T>::CUVector(int dimX, T* cpuBuf, T* gpuBuf) :
		_dimX(dimX), _dimY(1), _dimZ(1),
		_cpuData(cpuBuf), _gpuData(gpuBuf) { objInit(); }

// Funcion auxiliar para inicializar el objeto
template <typename T> void
CUVector<T>::objInit() {
	if(_dimX < 1 || _dimY < 1 || _dimZ < 1) raise("invalid size");
	if(!_cpuData) _cpuData = cpuAlloc(_dimX, _dimY, _dimZ);
	if(!_gpuData) _gpuData = gpuAlloc(_dimX, _dimY, _dimZ);
}

// Constructor (reserva memoria solamente en la CPU)
template <typename T> CUVector<T>*
CUVector<T>::CpuVector(int dimX, int dimY, int dimZ, T* cpuBuf) {
	if(dimX < 1 || dimY < 1 || dimZ < 1) raise("invalid size");
	CUVector* objPtr = new CUVector(dimX, dimY, dimZ, cpuBuf, (T*)-1);
	objPtr->_gpuData = NULL; // Cambiamos el (T*)-1 por NULL
	return objPtr;
}

// Constructor (reserva memoria solamente en la GPU)
template <typename T> CUVector<T>*
CUVector<T>::GpuVector(int dimX, int dimY, int dimZ, T* gpuBuf) {
	if(dimX < 1 || dimY < 1 || dimZ < 1) raise("invalid size");
	CUVector* objPtr = new CUVector(dimX, dimY, dimZ, (T*)-1, gpuBuf);
	objPtr->_cpuData = NULL; // Cambiamos el (T*)-1 por NULL
	return objPtr;
}

// Crea una instancia nueva basandose en un objeto ya existente
template <typename T> CUVector<T>*
CUVector<T>::instance() const {
	CUVector* objPtr;
	if(_gpuData && _cpuData) objPtr = CUVector(_dimX, _dimY, _dimZ);
	if(!_cpuData) objPtr = CUVector<T>::CpuVector(_dimX, _dimY, _dimZ);
	if(!_gpuData) objPtr = CUVector<T>::GpuVector(_dimX, _dimY, _dimZ);
	
	if(!objPtr) raise("invalid instance");
	return objPtr;
}

// Destructor, libera la memoria reservada en los dispositivos
template <typename T>
CUVector<T>::~CUVector() {
	if(_cpuData) cpuFree(_cpuData);
	if(_gpuData) gpuFree(_gpuData);
}

//---- Gestion de datos --------------------------------------------------

// Carga los datos en el vector de CPU, sin argumentos hace un commit
template <typename T> CUVector<T>*
CUVector<T>::cpuRead(const T* buf, int size) {
	if(!buf) buf = _gpuData;
	if(size != -1 && size != getSize()) raise("invalid size");
	if(!buf) raise("Invalid read buffer");
	if(!_cpuData) raise("not supported on GPU vector");
	return copy(_cpuData, buf, getSize(), -1); // UVA Caps 1.x: -1
}

// Carga los datos en el vector de GPU, sin argumentos hace un commit
template <typename T> CUVector<T>*
CUVector<T>::gpuRead(const T* buf, int size) {
	if(!buf) buf = _cpuData;
	if(size != -1 && size != getSize()) raise("invalid size");
	if(!buf) raise("Invalid read buffer");
	if(!_gpuData) raise("not supported on CPU vector");
	return copy(_gpuData, buf, getSize(),  1); // UVA Caps 1.x: 1
}

// Carga los datos de otro vector en CPU o en GPU
template <typename T> CUVector<T>*
CUVector<T>::copy(const CUVector<T>* src) {
	if((getX() != src->getX()) || (getY() != src->getY()) || (getZ() != src->getZ()))	
		return NULL;
		
	if(_cpuData && src->_cpuData) { // Vector CPU
		copy(_cpuData, src->_cpuData, getSize(), 1); // UVA Caps 1.x: 1
		if(_gpuData) gpuRead(); // Si tiene datos en GPU los actualizamos
	} else if(_gpuData && src->_gpuData) // Vector GPU
		copy(_gpuData, src->_gpuData, getSize(), 1); // UVA Caps 1.x: 1
	return this;
}

// Para depuracion, imprime informacion sobre el objeto
template <typename T> CUVector<T>*
CUVector<T>::debug(const char *name) const {
	if(!name) name = "<CUVector>";
	switch(getDim()) {
		case 1: printf("%10s[%4i] -> cpuPtr: %p, gpuPtr: %p\n", 
			name, _dimX, _cpuData, _gpuData); break;
		case 2: printf("%10s[%4i,%4i] -> cpuPtr: %p, gpuPtr: %p\n", 
			name, _dimX, _dimY, _cpuData, _gpuData); break;
		case 3: printf("%10s[%4i,%4i,%i] -> cpuPtr: %p, gpuPtr: %p\n", 
			name, _dimX, _dimY, _dimZ, _cpuData, _gpuData); break;
	}
	return const_cast<CUVector<T>*>(this);
}

// Muestra informacion sobre los errores de cuda
template <typename T> void
CUVector<T>::error(const char* errMsg, int errCode) {
	if(errCode == 0) errCode = cudaGetLastError();
	if(!errCode) return;
	if(!errMsg) errMsg = cudaGetErrorString((cudaError_t)errCode);
	printf("\nError %i: %s\n", errCode, errMsg); exit(0);
}

// Muestra informacion sobre los errores de cuda
template <typename T> void
CUVector<T>::error(const char* fName, const char* errMsg, int errCode) {
	printf("\n--- File '%s' ---", fName);
	error(errMsg, errCode);
}

// Usar para sincronizar la ejecucion y poder tomar tiempos
template <typename T> void
CUVector<T>::sync() {
	cudaError_t errCode = cudaDeviceSynchronize();
	if(errCode != cudaSuccess) raise("cuda sync");
}

// Establece el dispositivo actual cuando hay varias GPUs
template <typename T> void
CUVector<T>::gpuInit(int devId) {
	// Comprobamos si ya existian errores previos
	cudaError_t errCode = cudaGetLastError();
	if(errCode != cudaSuccess) goto onError;

	// Comprobamos cuantos dispositivos hay disponibles
	int numDev, drvVer, runVer;
	if((errCode = cudaGetDeviceCount(&numDev)) != 0) goto onError;
	if(numDev == 0) {
		printf("gpuInit: No compatible CUDA devices found");
		exit(0);
	}
	
	cudaDriverGetVersion(&drvVer); cudaRuntimeGetVersion(&runVer);
	printf("gpuInit : Runtime v%i, Driver v%i\n", runVer, drvVer);
	printf("gpuInit : Found %i compatible devices\n", numDev);

	// Mostramos informaci髇 de los dispositivos
	cudaDeviceProp devInfo; // TODO: Soporte Multi-GPU
	for(int dev = 0; dev < numDev; dev++) {
		if((errCode = cudaGetDeviceProperties(&devInfo, dev)) != 0) goto onError;
		long long memory = (long long)devInfo.totalGlobalMem;
		printf("gpuInit : Device %i : %s, %lli MB\n",
			dev, devInfo.name, memory >> 20);
	}
	// Establecemos el dispositivo seleccionado
	if((errCode = cudaSetDevice(devId < numDev ? devId : 0)) != 0) goto onError;
	if((errCode = cudaGetDevice(&devId)) != 0) goto onError;
	printf("gpuInit : Using device %i for GPGPU\n", devId);

onError:
	// Verificamos si ocurrio un error e imprimimos la informacion
	if(errCode == cudaSuccess) return;
	const char* errMsg = cudaGetErrorString(errCode);
	printf("gpuInit : Error while initializing GPU\n");
	printf("gpuInit : '%s'\n", errMsg);
	exit(0);
}

//---- Metodos extendidos ------------------------------------------------

// Aplica una funcion (independiente de la posicion)
template <typename T> CUVector<T>*
CUVector<T>::map(void (*mapFun)(T& obj)) {
	if(mapFun == NULL) raise("invalid mapping function");
	if(!_cpuData) raise("not supported on GPU vector");

	const int size = getSize();
	for(int i = 0; i < size; i++)
		mapFun(_cpuData[i]);
	return this;
}

// Aplica una funcion (dependiente de la posicion 1D)
template <typename T> CUVector<T>*
CUVector<T>::map(void (*mapFun)(T& obj, int pos)) {
	if(mapFun == NULL) raise("invalid mapping function");
	if(!_cpuData) raise("not supported on GPU vector");

	const int size = getSize();
	for(int i = 0; i < size; i++)
		mapFun(_cpuData[i], i);
	return this;
}

// Aplica una funcion (dependiente de la posicion 2D)
template <typename T> CUVector<T>*
CUVector<T>::map(void (*mapFun)(T& obj, int posX, int posY)) {
	if(mapFun == NULL) raise("invalid mapping function");
	if(!_cpuData) raise("not supported on GPU vector");

	for(int y = 0; y < _dimY; y++) {
		const int base = y * _dimX;
		for(int x = 0; x < _dimX; x++)
			mapFun(_cpuData[base + x], x, y);
	}
	return this;
}

// Aplica una funcion (dependiente de la posicion 3D)
template <typename T> CUVector<T>*
CUVector<T>::map(void (*mapFun)(T& obj, int posX, int posY, int posZ)) {
	if(mapFun == NULL) raise("invalid mapping function");
	if(!_cpuData) raise("not supported on GPU vector");

	for(int z = 0; z < _dimZ; z++)
		for(int y = 0; y < _dimY; y++) {
			const int base = y * _dimX + z * _dimX * _dimY;
			for(int x = 0; x < _dimX; x++)
				mapFun(_cpuData[base + x], x, y, z);
		}
	return this;
}

// Aplica una funcion de forma transpuesta yxz (dependiente de la posicion 3D)
template <typename T> CUVector<T>*
CUVector<T>::mapT(void (*mapFun)(T& obj, int posX, int posY, int posZ)) {
	if(mapFun == NULL) raise("invalid mapping function");
	if(!_cpuData) raise("not supported on GPU vector");

	for(int z = 0; z < _dimZ; z++)
		for(int x = 0; x < _dimX; x++) {
			const int base = x + z * _dimX * _dimY;
			for(int y = 0; y < _dimY; y++)
				mapFun(_cpuData[base + y * _dimX], x, y, z);
		}
	return this;
}

// Aplica una reduccion (independiente de la posicion)
template <typename T> CUVector<T>*
CUVector<T>::reduce(void (*redFun)(T& redVar, const T& obj), T& redVar) const {
	if(redFun == NULL) raise("invalid reduce function");
	if(!_cpuData) raise("not supported on GPU vector");

	const int size = getSize();
	for(int i = 0; i < size; i++)
		redFun(redVar, _cpuData[i]);
	return const_cast<CUVector<T>*>(this);
}

//---- Metodos privados --------------------------------------------------

template <typename T> CUVector<T>*
CUVector<T>::copy(T* dst, const T* src, int dim, int uva) const {
	if(!dst) raise("invalid destination buffer");
	if(!src) raise("invalid source buffer");
	if(dim < 1) raise("invalid size");
	if(uva < -1 || uva > 1) raise("check uva memory space");

	size_t nBytes = dim * sizeof(T);
	cudaMemcpyKind cpyFlag = cudaMemcpyDefault;
	#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
		if(uva ==  1) cpyFlag = cudaMemcpyHostToDevice; // Load
		if(uva == -1) cpyFlag = cudaMemcpyDeviceToHost; // Unload
	#endif
	cudaError_t errCode = cudaMemcpy(dst, src, nBytes, cpyFlag);
	if(errCode != cudaSuccess) raise("buffer copy");
	return const_cast<CUVector<T>*>(this);
}

// Reservar pinned memoria en la CPU para mejorar transferencias
template <typename T> T*
CUVector<T>::cpuAlloc(int dimX, int dimY, int dimZ) const {
	size_t nBytes = dimX * dimY * dimZ * sizeof(T);
	if(nBytes <= 0) raise("invalid size");

	T* cpuPtr = NULL;
	unsigned int flags = cudaHostAllocDefault;
	cudaError_t errCode = cudaHostAlloc((void**)&cpuPtr, nBytes, flags);
	//x printf("cpuAlloc %i bytes (%i MB): %p\n", nBytes, nBytes/(1<<20), cpuPtr);
	if(errCode != cudaSuccess) raise("cpu allocation error");
	return cpuPtr;
}

// Liberar memoria de la GPU
template <typename T> void
CUVector<T>::cpuFree(T* cpuPtr) const {
	if(cpuPtr == NULL) raise("null pointer");
	cudaError_t errCode = cudaFreeHost(cpuPtr);
	//x printf("cpuFree: %p\n", cpuPtr);
	if(errCode != cudaSuccess) raise("invalid pointer");
}

// Reservar memoria en la GPU usando el API de CUDA
template <typename T> T*
CUVector<T>::gpuAlloc(int dimX, int dimY, int dimZ) const {
	size_t nBytes = dimX * dimY * dimZ * sizeof(T);
	if(nBytes <= 0) raise("invalid size");

	T* gpuPtr = NULL;
	cudaError_t errCode = cudaMalloc((void**)&gpuPtr, nBytes);
	//x printf("gpuAlloc %i bytes (%i MB): %p\n", nBytes, nBytes/(1<<20), gpuPtr);
	if(errCode != cudaSuccess) raise("gpu allocation error");
	return gpuPtr;
}

// Liberar memoria de la GPU
template <typename T> void
CUVector<T>::gpuFree(T* gpuPtr) const {
	if(gpuPtr == NULL) raise("null pointer");
	cudaError_t errCode = cudaFree(gpuPtr);
	//x printf("gpuFree: %p\n", gpuPtr);
	if(errCode != cudaSuccess) raise("invalid pointer");
}

#endif // CUVector_CPP

