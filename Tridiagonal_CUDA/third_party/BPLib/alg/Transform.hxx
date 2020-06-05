//- =======================================================================
//+ Transforms interface
//- =======================================================================

#pragma once
#ifndef TRANSFORM_HXX
#define TRANSFORM_HXX

//---- Header Dependencies -----------------------------------------------

#include "tools/cuvector.hxx"

//---- Class Declaration -------------------------------------------------

class Transform {

public:
	
	// Constructor by default
	Transform() { }

	// Creates a transform, batch is the last dimension
	Transform(int dimX, int dimY = 1, int dimZ = 1);

	// Free resources
	virtual ~Transform();
	
	// Calculates transform, direct (dir=1) or inverse (dir=-1)
	virtual Transform* calc(int dir = 1) = 0;

	// Initializes data, zero (mode=-1), seq (mode=0) or rand (mode=1)
	virtual Transform* init(int mode = -1) = 0;

	// Initializes data by copying from other transform
	virtual Transform* init(const Transform* src);

	// Printing data
	virtual Transform* print(const char* name = 0) = 0;

	virtual const char* toString() = 0;

	// Compares error with respect to other algorithm
	virtual double compare(Transform* ref, const char* label = 0) = 0;

	virtual double gflops(double time, long long iters = 1) const = 0;

	friend class CudaFourier;
	
	// Updates vector data in CPU
	inline void updateCpu() { updateCpu_(); updateCpu_r(); updateCpu_c(); }

	// Updates vector data in GPU
	inline void updateGpu() { updateGpu_(); updateGpu_r(); updateGpu_c(); }
	
	

protected:

	// Used by compare in order to obtain the integer data vector pointer
	int * getData_() const {return _dat ? _dat->cpuData(): NULL;}

	// Used by compare in order to obtain the real data vector pointer
	float* getData_r() const { return _rdat ? _rdat->cpuData() : NULL; }
	
	// Used by compare in order to obtain the complex data vector pointer
	Complex* getData_c() const { return _cdat ? _cdat->cpuData() : NULL; }

	// Used by GPU algorithms for accesing to integer vector in RAW mode
	inline CUVector<int>* gpuVector_() const { return _dat; }

	// Used by GPU algorithms for accesing to complex vector in RAW mode
	inline CUVector<Complex>* gpuVector_c() const { return _cdat; }

	// Used by GPU algorithms for accesing to real vector in RAW mode
	inline CUVector<float>* gpuVector_r() const { return _rdat; }

	// Updates integer data in CPU
	inline void updateCpu_() { if(_dat) _dat->cpuRead(); }	
	
	// Updates real data in CPU
	inline void updateCpu_r() { if(_rdat) _rdat->cpuRead(); }

	// Updates complex data in CPU
	inline void updateCpu_c() { if(_cdat) _cdat->cpuRead(); }

	// Updates integer data in GPU
	inline void updateGpu_() { if(_dat) _dat->gpuRead(); }

	// Updates real data in GPU
	inline void updateGpu_r() { if(_rdat) _rdat->gpuRead(); }

	// Updates complex data in GPU
	inline void updateGpu_c() { if(_cdat) _cdat->gpuRead(); }


	// Initializes integer data ( if device = 0 it does not allocate on GPU)
	void init_(int mode = -1, int device = 1);

	// Initializes real data ( if device = 0 it does not allocate on GPU)
	void init_r(int mode = -1, int device = 1);

	// Initializes complex data ( if device = 0 it does not allocate on GPU)
	void init_c(int mode = -1, int device = 1);

	//Prints integer data
	void print_(const char* name) const;

	// Prints real data
	void print_r(const char* name) const;

	// Prints complex data
	void print_c(const char* name) const;

	// Compares integer data
	float compare_(const Transform* ref, const char* label) const;

	// Compares real data
	float compare_r(const Transform* ref, const char* label) const;

	// Compares complex data
	float compare_c(const Transform* ref, const char* label) const;

	// Returns gflops over integer data
	double gflops_(double time, long long iters = 1) const;

	// Returns gflops over real data
	double gflops_r(double time, long long iters = 1) const;

	// Returns gflops over complex data
	double gflops_c(double time, long long iters = 1) const;

	// X dimension specified by constructor
	inline int getX() const { return _dimX; }

	// Y dimension specified by constructor
	inline int getY() const { return _dimY; }

	// Z dimension specified by constructor
	inline int getZ() const { return _dimZ; }

	// PI value (inline)
	static inline const float pi_32() { return 3.141592653f; }

	// PI value (inline)
	static inline const double pi_64() { return 3.141592653589793238; }

private:

	
	// Initializes to 0 the integer parameter
	static void mapClr(int & val, int x, int y, int z) { val = 0 *(x*y*z); }

	// Initializes to 1 the integer parameter
	static void map1(int& val, int x, int y, int z) { val = 1 ; }
	
	// Generates an integer list of a linear function
	static void mapLin(int& val, int x, int y, int z);

	// Generates a sequencial list of integers
	static void mapSeq(int& val, int x, int y, int z);

	// Generates a random sequence of integer values
	static void mapRnd(int& val, int x, int y, int z);

	// Prints the value and position of an integer value
	static void mapPut(int& val, int posX, int posY, int posZ);

	// Initializes an integer value to the unity
	static void mapOne(int& val, int x, int y, int z);

	// Prints an integer tridiagonal system
	static void mapTri(int& val, int posX, int posY, int posZ);

	// Prints integer values in power of two
	static void mapLog(int& val, int posX, int posY, int posZ);

	// Initializes to 0 the real parameter
	static void mapClr(float& val, int x, int y, int z) { val = 0.0f *(x*y*z); }

	// Initializes to 1 the real parameter
	static void map1(float& val, int x, int y, int z) { val = 1.0f ; }
	

	// Generates a real list of a linear function
	static void mapLin(float& val, int x, int y, int z);

	// Generates a sequence of real values
	static void mapSeq(float& val, int x, int y, int z);

	// Generates a random sequence of real values
	static void mapRnd(float& val, int x, int y, int z);

	// Prints the value and position of a real value
	static void mapPut(float& val, int posX, int posY, int posZ);

	// Initializes a real value to the unity
	static void mapOne(float& val, int x, int y, int z);

	// Prints a real tridiagonal system
	static void mapTri(float& val, int posX, int posY, int posZ);

	// Prints real values in power of two
	static void mapLog(float& val, int posX, int posY, int posZ);

	// Initializes to 0 the complex parameter
	static void mapClr(Complex& val, int x, int y, int z) { val = 0.0f *(x*y*z); }

	// Generates a complex list of a linear function
	static void mapLin(Complex& val, int x, int y, int z);

	// Generates a sequence of complex values
	static void mapSeq(Complex& val, int x, int y, int z);

	// Generates a random sequence of complex values
	static void mapRnd(Complex& val, int x, int y, int z);

	// Initializes a complex value to the unity
	static void mapOne(Complex& val, int x, int y, int z);

	// Prints the value and position of a complex value
	static void mapPut(Complex& val, int posX, int posY, int posZ);

	// Prints a complex tridiagonal system
	static void mapTri(Complex& val, int posX, int posY, int posZ);

	// Prints complex value in power of two
	static void mapLog(Complex& val, int posX, int posY, int posZ);

	// Compares two integers, returning the abs()
	static float comp(const int& a, const int& b);

	// Compares two reals, returning the abs()
	static float comp(const float& a, const float& b);

	// Compares two complex, returning the abs()
	static float comp(const Complex& a, const Complex& b);

	// Pointer to integer data
	CUVector<int>* _dat;

	// Pointer to real data
	CUVector<float>*   _rdat;

	// Pointer to complex data
	CUVector<Complex>* _cdat;

	// X,Y and Z dimensions
	int _dimX, _dimY, _dimZ;

	// Additionally flags
	int _flags;
};

#endif // TRANSFORM_HXX

