//- =======================================================================
//+ Cuda Vector Definition v3.0
//- =======================================================================

#pragma once
#ifndef CUVector_HXX
#define CUVector_HXX

//---- Library Macro Definitions -----------------------------------------


#include <complex>
typedef std::complex<float> Complex;
typedef std::complex<double> Complex64;

inline Complex64 C32toC64(const Complex& val) {
	return Complex64((double)val.real(), (double)val.imag());
}

inline Complex C64toC32(const Complex64& val) {
	return Complex((float)val.real(), (float)val.imag());
}


#ifndef _RAISE_
	#define _RAISE_
	#define raise(_str) error(__FUNCTION__, _str, __LINE__)
#endif // _RAISE_

//---- Library Header Definitions ----------------------------------------

template<typename T> class CUVector {

public:

	// Constructores (reserva memoria tanto en la CPU como en la GPU)
	CUVector(int dimX, T* cpuBuf = 0, T* gpuBuf = 0);
	CUVector(int dimX, int dimY, T* cpuBuf = 0, T* gpuBuf = 0);
	CUVector(int dimX, int dimY, int dimZ, T* cpuBuf = 0, T* gpuBuf = 0);

	// Constructor (reserva memoria solamente en la CPU)
	inline static CUVector<T>* CpuVector(int dimX, T* cpuBuf = 0) {
		return CpuVector(dimX, 1, 1, cpuBuf); }
	inline static CUVector<T>* CpuVector(int dimX, int dimY, T* cpuBuf = 0) {
		return CpuVector(dimX, dimY, 1, cpuBuf); }
	static CUVector<T>* CpuVector(int dimX, int dimY, int dimZ, T* cpuBuf = 0);
	
	// Constructor (reserva memoria solamente en la GPU)
	inline static CUVector<T>* GpuVector(int dimX, T* gpuBuf = 0) {
		return GpuVector(dimX, 1, 1, gpuBuf); }
	inline static CUVector<T>* GpuVector(int dimX, int dimY, T* gpuBuf = 0) {
		return GpuVector(dimX, dimY, 1, gpuBuf); }
	static CUVector<T>* GpuVector(int dimX, int dimY, int dimZ, T* gpuBuf = 0);

	// Crea una instancia nueva basandose en un objeto ya existente
	CUVector* instance() const;

	// Destructor, libera la memoria reservada en los dispositivos
	~CUVector();

	// Carga los datos en el vector de CPU, sin argumentos hace un commit
	CUVector<T>* cpuRead(const T* buf = NULL, int size = -1);

	// Carga los datos en el vector de GPU, sin argumentos hace un commit
	CUVector<T>* gpuRead(const T* buf = NULL, int size = -1);

	// Carga los datos de otro vector
	CUVector<T>* copy(const CUVector<T>* src);

	// Para depuracion, imprime informacion sobre el objeto
	CUVector<T>* debug(const char *name = 0) const;

	// Muestra informacion sobre los errores de cuda
	static void error(const char* errMsg = 0, int errCode = 0);

	// Muestra informacion sobre los errores de cuda
	static void error(const char* fName, const char* errMsg, int errCode = 0);

	// Usar para sincronizar la ejecucion y poder tomar tiempos
	static void sync();

	// Establece el dispositivo actual cuando hay varias GPUs
	static void gpuInit(int devId = 0);

	// Hace un cast automatico a los datos almacenados en la CPU
	operator T*() const { return _cpuData; }

	// Permite acceder a los datos de CPU como vector 1D
	// inline T& operator[](int pos) const { return _cpuData[pos]; }

	// Devuelve un puntero al principio de los datos en CPU
	inline T* begin() const { return _cpuData; }

	// Devuelve un puntero al final de los datos en GPU
	inline T* end() const { return _cpuData + getSize(); }

	// Aplica una funcion (que puede depender de la posicion)
	CUVector<T>* map(void (*mapFun)(T& obj));
	CUVector<T>* map(void (*mapFun)(T& obj, int pos));
	CUVector<T>* map(void (*mapFun)(T& obj, int posX, int posY));
	CUVector<T>* map(void (*mapFun)(T& obj, int posX, int posY, int posZ));

	// Aplica la funcion, pero transponiendo las coordenadas X e Y
	CUVector<T>* mapT(void(*mapFun)(T& obj, int posX, int posY, int posZ));

	// Aplica una reduccion (independiente de la posicion)
	CUVector<T>* reduce(void (*redFun)(T& redVar, const T& obj), T& redVar) const;

protected:
	
	// Tama駉 del vector
	int _dimX, _dimY, _dimZ;

	// Puntero a los datos en los dispositivos
	T *_cpuData, *_gpuData;

public:

	// Devuelve el tama駉 del array en GPU
	inline int getSize() const { return _dimX * _dimY * _dimZ; }

	// Devuelve la primera dimension
	inline int getX() const { return _dimX; }

	// Devuelve la segunda dimension
	inline int getY() const { return _dimY; }

	// Devuelve la tercera dimension
	inline int getZ() const { return _dimZ; }

	// Devuelve la segunda dimension
	inline int getDim() const { return 1 + (_dimY > 1) + (_dimZ > 1); }

	// Devuelve un puntero a los datos en CPU
	inline T* cpuData() const { return _cpuData; };

	// Devuelve un puntero a los datos en GPU
	inline T* gpuData() const { return _gpuData; }

	// Permite acceder a los datos de CPU como vector 1D, 2D o 3D
	inline T& operator()(int x) const {
		return _cpuData[x]; }
	inline T& operator()(int x, int y) const {
		return _cpuData[x + y * _dimX]; }
	inline T& operator()(int x, int y, int z) const {
		return _cpuData[x + y * _dimX + z * _dimX * _dimY]; }

private:

	// Funcion auxiliar para inicializar el objeto
	void objInit();

	// Copia los datos entre punteros a memoria
	CUVector<T>* copy(T* dst, const T* src, int dim, int uva = 0) const;

	// Reservar memoria en la CPU
	T* cpuAlloc(int dimX, int dimY = 1, int dimZ = 1) const;

	// Reservar memoria en la GPU
	T* gpuAlloc(int dimX, int dimY = 1, int dimZ = 1) const;

	// Liberar memoria de la CPU
	void cpuFree(T* cpuPtr) const;

	// Liberar memoria de la GPU
	void gpuFree(T* gpuPtr) const;

};

//---- Header Dependencies -----------------------------------------------
#include "cuvector.cpp"

#endif // CUVector_HXX

