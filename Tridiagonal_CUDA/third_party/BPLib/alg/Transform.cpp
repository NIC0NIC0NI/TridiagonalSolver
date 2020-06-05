//- =======================================================================
//+ Transform interface
//- =======================================================================

//---- Header Dependencies -----------------------------------------------
#include "Transform.hxx"
#include "tools/tausgen.hxx"
#include <cstring>

//---- Library Header Definitions ----------------------------------------


Transform::Transform(int dimX, int dimY, int dimZ) :
	_dat(0), _rdat(0), _cdat(0), _dimX(dimX), _dimY(dimY), _dimZ(dimZ) { }


Transform::~Transform() {
	if(_dat) delete _dat;
	if(_rdat) delete _rdat;
	if(_cdat) delete _cdat;
}

Transform* Transform::init(const Transform *src) {

	

	int errCode = 0;

	if(_dat && src->_dat)
		errCode += (_dat->copy(src->_dat) == NULL);

	if(_rdat && src->_rdat)
		errCode += _rdat->copy(src->_rdat) == NULL;

	if(_cdat && src->_cdat)
		errCode += _cdat->copy(src->_cdat) == NULL;

	

	return errCode ? 0 : this;
}





//--------------------------------- Protected Methods ------------------------------------------------


void Transform::init_(int mode, int device)
{

	

	if(!_dat) {
		if(device == 1) _dat = new CUVector<int>(_dimX, _dimY, _dimZ);
		else _dat = CUVector<int>::CpuVector(_dimX, _dimY, _dimZ);

	
	}
	switch(mode) {
		case 0 : _dat->map(mapLin); break;
		case 1 : _dat->map(mapRnd); break;
		case 2 : _dat->map(mapSeq); break;
		case 3 : _dat->map(mapOne); break;
		case 4 : _dat->map(map1);   break;
		default: _dat->map(mapClr); break;
	}

	

}




void Transform::init_r(int mode, int device) {
	if(!_rdat) {
		if(device == 1) _rdat = new CUVector<float>(_dimX, _dimY, _dimZ);
		else _rdat = CUVector<float>::CpuVector(_dimX, _dimY, _dimZ);
	}
	switch(mode) {
		case -1: break;
		case 0 : _rdat->map(mapLin); break;
		case 1 : _rdat->map(mapRnd); break;
		case 2 : _rdat->map(mapSeq); break;
		case 3 : _rdat->map(mapOne); break;
		case 4 : _rdat->map(map1);   break;
		default: _rdat->map(mapClr); break;
	}
}






void Transform::init_c(int mode, int device) {
	if(!_cdat) {
		if(device == 1) _cdat = new CUVector<Complex>(_dimX, _dimY, _dimZ);
		else _cdat = CUVector<Complex>::CpuVector(_dimX, _dimY, _dimZ);
	}
	switch(mode) {
		case 0 : _cdat->map(mapLin); break;
		case 1 : _cdat->map(mapRnd); break;
		case 2 : _cdat->map(mapSeq); break;
		case 3 : _cdat->map(mapOne); break;
		default: _cdat->map(mapClr); break;
	}
}


void Transform::print_(const char* name) const {
	if(!name) name = "<unnamed transform>";
	if(!_dat) CUVector<int>::raise("Null pointer");
	
	printf("%-10s : {%c\n", name, strlen(name) > 10 ? ' ' : '\t');
	
	int isToLong = getX() * getY() >= 4096;
	if(isToLong) _dat->map(mapLog);
	else _dat->map(mapPut);
	
	printf("}\n");
}


void Transform::print_r(const char* name) const {
	if(!name) name = "<unnamed transform>";
	if(!_rdat) CUVector<float>::raise("Null pointer");
	int isTridiag = getY() == 4 && getX() >= 3;
	printf("%-10s : {%c%s", name, strlen(name) > 10 ? ' ' : '\t',
		isTridiag ? "--dl--\t\t--d--\t\t--du--\t\t--x--\n" : "\n");
	if(isTridiag) {
		_rdat->mapT(mapTri);
	} else {
		int isToLong = getX() * getY() >= 4096;
		if(isToLong) _rdat->map(mapLog);
		else _rdat->map(mapPut);
	}
	printf("}\n");
}


void Transform::print_c(const char* name) const {
	if(!name) name = "<unnamed transform>";
	if(!_cdat) CUVector<Complex>::raise("Null pointer");
	int isTridiag = getY() == 4 && getX() >= 3;
	printf("%-10s : {%c%s", name, strlen(name) > 10 ? ' ' : '\t', isTridiag ?
		"    --- dl ---\t\t    --- d ---\t\t    --- du ---\t\t    --- x ---\n" : "\n");

	if(isTridiag) {
		_cdat->mapT(mapTri);
	} else {
		int isToLong = getX() * getY() >= 4096;
		if(isToLong) _cdat->map(mapLog);
		else _cdat->map(mapPut);
	}
	printf("}\n");
}


float Transform::compare_c(const Transform* ref, const char* label) const {
	// Verify input parameters
	if(!ref) CUVector<Complex>::raise("Invalid argument");
	const Complex* s1 = _cdat->cpuData();
	const Complex* s2 = ref->_cdat ? ref->_cdat->cpuData() : NULL;
	if(!s1 || !s2) CUVector<Complex>::raise("Invalid comparison (R/C?)");
	const int size = _cdat->getSize();
	if(size != ref->_cdat->getSize())
		CUVector<Complex>::raise("Length does not match");

	// Calculate error between vectors
	double accError = 0.0f;
	static double maxError = 1e-3f;
	for(int i = 0; i < size; i++) {
		double relError = comp(s1[i], s2[i]);
		accError += relError;
		if(relError <= maxError) continue;
		maxError = 2 * relError;
		printf("Warning (DIF): Element %i relError is %f\n\t%f%+fi != % f%+fi\n",
			i, relError, s1[i].real(), s1[i].imag(), s2[i].real(), s2[i].imag());
	}

	// Printing the label and returning the relative error
	float meanError = (float)accError / size;
	if(label && meanError > 0.1f) printf("in test: %s\n", label);
	return meanError;
}


float Transform::compare_r(const Transform* ref, const char* label) const {
	// Verify input parameters
	if(!ref) CUVector<float>::raise("Invalid argument");
	const float* s1 = _rdat ? _rdat->cpuData() : NULL;
	const float* s2 = ref->_rdat ? ref->_rdat->cpuData() : NULL;
	if(!s1 || !s2) CUVector<float>::raise("Invalid comparison (R/C?)");
	const int size = _rdat->getSize();
	if(size != ref->_rdat->getSize())
		CUVector<float>::raise("Length does not match");

	

	// Calculate error between vectors
	double accError = 0.0f;
	static double maxError = 0.1f;
	for(int i = 0; i < size; i++) {		
		//printf("%d -> %lf \t vs \t %lf \n",i,s1[i],s2[i]);		
		double relError = comp(s1[i], s2[i]);
		accError += relError;
		if(relError <= maxError) continue;
		maxError = 2 * relError;
		printf("Warning (DIF): Element %i relError is %f\n\t%f != %f\n",
			i, relError, s1[i], s2[i]);
	}

	// Printing the label and returning the relative error
	float meanError = (float)accError / size;
	if(label && meanError > 0.1f) printf("in test: %s\n", label);
	return meanError;
}



float Transform::compare_(const Transform* ref, const char* label) const {
	// Verify the input parameters
	if(!ref) CUVector<int>::raise("Invalid argument");
	const int* s1 = _dat ? _dat->cpuData() : NULL;
	const int* s2 = ref->_dat ? ref->_dat->cpuData() : NULL;
	if(!s1 || !s2) CUVector<int>::raise("Invalid comparison (R/C?)");
	const int size = _dat->getSize();
	if(size != ref->_dat->getSize())
		CUVector<int>::raise("Length does not match");

	// Calculate error between vectors
	double accError = 0.0f;
	static double maxError = 1e-3f;
	for(int i = 0; i < size; i++) {
		double relError = comp(s1[i], s2[i]);
		accError += relError;
		if(relError <= maxError) continue;
		maxError = 2 * relError;
		printf("Warning (DIF): Element %i relError is %f\n\t%d != %d\n",
			i, relError, s1[i], s2[i]);
	}

	// Printing the label and returning the relative error
	float meanError = (float)accError / size;
	if(label && meanError > 0.1f) printf("in test: %s\n", label);
	return meanError;
}


double Transform::gflops_c(double time, long long iters) const {
	double numFFTs = (double)iters * (double)_dimZ;
	double log2x = log((double)_dimX) / log(2.0);
	double log2y = log((double)_dimY) / log(2.0);
	double nFlops = numFFTs * 5 * _dimX * _dimY * (log2x + log2y);
	if(time < 1e-3) time = 1e-3; // Prevent division by zero
	return nFlops * 1e-9 / time;	
}

double Transform::gflops_r(double time, long long iters) const {
	return 0.5 * gflops_c(time, iters);
}


double Transform::gflops_(double time, long long iters) const {
	return 0.5 * gflops_c(time, iters);
}









//-------------------------------------- Private --------------------------------------------------


void Transform::mapLin(int& val, int x, int y, int z) {
	val = 1 * x + y * 1 + z * 1;
}


void Transform::mapSeq(int& val, int x, int y, int z) {
	static int cont = 0;
	cont = x+y+z ? cont+1 : 0;

	
	val =  cont ;
}



void Transform::mapRnd(int & val, int x, int y, int z) {
	if(x == 0 && y == 0 && z == 0) tausInit(0);
	const int mask = 0x1F;
	const int desp = 1 * (mask >> 1);
	val = (tausRand() & mask) - desp;
}



void Transform::mapOne(int& val, int x, int y, int z) {
	val = 1 * (y&0x01);
	/*int ini[4] = { 2.0f, 9.0f, 3.0f, 1e3f };
	val = ini[y & 0x03];
	*/
	/*float ini[4] = { 1.0f, 100.0f, 1.0f, 1e3f };
	if(!(x & 0x0fff) && !(y & 0x0fff)) srand(0);
	val = ini[y & 0x03] * (1 + (rand() & 0x07f));
	*/
	/*
	float ini[4] = { 10.0f, 1.0f, 10.0f, 10.0f };
	if(!(x & 0x0fff) && !(y & 0x0fff)) tausInit(0);
		val = ini[y & 0x03] * tausRand(1,99);
		*/
	/*
	float ini[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
	if(!(x & 0x0fff) && !(y & 0x0fff)) tausInit(0);
		val = ini[y & 0x03] * tausRand(1,99);
		*/
}


void Transform::mapPut(int& val, int posX, int posY, int posZ) {
	printf("(%3i,%3i)[%3i] : %d\n",
		posX, posY, posZ, val);
}


void Transform::mapTri(int& val, int posX, int posY, int posZ) {
	if(posY) printf("%d %c", val, posY > 2 ? '\n' : '\t');
	else     printf("(%3i)[%3i] %c \t %d \t", posY, posZ,
		posX ? ':' : '#', val);
}




void Transform::mapLin(float& val, int x, int y, int z) {
	val = 1.0f * x + y * 0.01f + z * 0.0001f;
}


void Transform::mapSeq(float& val, int x, int y, int z) {
	static int cont = 0;
	cont = x+y+z ? cont+1 : 0;
	val = 1.0f * cont * cont;
}


void Transform::mapRnd(float& val, int x, int y, int z) {
	if(x == 0 && y == 0 && z == 0) tausInit(0);
	const int mask = 0x1F;
	const float desp = 1.0f * (mask >> 1);
	val = (tausRand() & mask) - desp;
}





void Transform::mapOne(float& val, int x, int y, int z) {
	//val = 1.0f * (y&0x01);
	//float ini[4] = { 2.0f, 9.0f, 3.0f, 1e3f };
	//val = ini[y & 0x03];
	
	float ini[4] = { 1.0f, 100.0f, 1.0f, 1e3f };
	if(!(x & 0x0fff) && !(y & 0x0fff)) srand(0);
	val = ini[y & 0x03] * (1 + (rand() & 0x07f));
	
	/*
	float ini[4] = { 10.0f, 1.0f, 10.0f, 10.0f };
	if(!(x & 0x0fff) && !(y & 0x0fff)) tausInit(0);
		val = ini[y & 0x03] * tausRand(1,99);
		*/
	/*
	float ini[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
	if(!(x & 0x0fff) && !(y & 0x0fff)) tausInit(0);
		val = ini[y & 0x03] * tausRand(1,99);
		*/
}


void Transform::mapPut(float& val, int posX, int posY, int posZ) {
	printf("(%3i,%3i)[%3i] : % 7.2f\n",
		posX, posY, posZ, val);
}


void Transform::mapTri(float& val, int posX, int posY, int posZ) {
	if(posY) printf("% 7.2f %c", val, posY > 2 ? '\n' : '\t');
	else     printf("(%3i)[%3i] %c \t % 7.2f \t", posY, posZ,
		posX ? ':' : '#', val);
}


void Transform::mapTri(Complex& val, int posX, int posY, int posZ) {
	if(posY) printf("% 7.2f, % 7.2f %c", 
				val.real(), val.imag(), posY > 2 ? '\n' : '\t');
	else     printf("(%3i)[%3i] %c\t% 7.2f, % 7.2f \t", posY, posZ,
		posX ? ':' : '#', val.real(), val.imag());
}


void Transform::mapLin(Complex& val, int x, int y, int z) {
	val = Complex(1.0f * x + z * 0.01f, 1.0f * y + z * z * 0.01f);
}


void Transform::mapSeq(Complex& val, int x, int y, int z) {
	val = Complex(1.0f * x, 1.0f * x * x);
}


void Transform::mapOne(Complex& val, int x, int y, int z) {
	val = Complex(1.0f * (y&0x01), 0.0f);
	//val = (y == 3) ? Complex(1.0f,0.0f) : Complex(3.0f,0.0f);
}


void Transform::mapRnd(Complex& val, int x, int y, int z) {
	if(x == 0 && y == 0 && z == 0) tausInit(0);
	const int mask = 0x0F;
	const float desp = 1.0f * (mask >> 1);
	float value1 = (tausRand() & mask) - desp;
	float value2 = (tausRand() & mask) - desp;
	val = Complex(value1, value2);
}


void Transform::mapPut(Complex& val, int posX, int posY, int posZ) {
	printf("(%3i,%3i)[%3i] : % 7.2f, % 7.2f\n",
		posX, posY, posZ, val.real(), val.imag());
}

float Transform::comp(const int& a, const int& b) {
	int abs_b = abs(b);
	return abs_b > 0 ? abs(a - b) / abs_b : 0.0f;
}


float Transform::comp(const float& a, const float& b) {
	float abs_b = fabs(b);
	return abs_b > 0.01f ? fabs(a - b) / abs_b : 0.0f;
}


float Transform::comp(const Complex& a, const Complex& b) {
	//return comp(abs(a), abs(b));
	return comp(a.real(), b.real()) + comp(a.imag(), b.imag());
}

void Transform::mapLog(Complex& val, int posX, int posY, int posZ) {
	if( (posX & (posX - 1)) || (posY & (posY - 1)) ) return;
	float fixVal = abs(val.real()) < 1e-3f ? 0.0f : abs(val.real());
	float log2val = log(fixVal) / log(2.0f);
	printf("(%7i,%7i)[%3i] : (%5.2f) % 10.2f, % 10.2f\n",
		posX, posY, posZ, log2val, val.real(), val.imag());
}

void Transform::mapLog(float& val, int posX, int posY, int posZ) {
	if( (posX & (posX - 1)) || (posY & (posY - 1)) ) return;
	printf("(%7i,%7i)[%3i] : % 10.2f\n",
		posX, posY, posZ, val);
}
void Transform::mapLog(int& val, int posX, int posY, int posZ) {
	if( (posX & (posX - 1)) || (posY & (posY - 1)) ) return;
	printf("(%7i,%7i)[%3i] : %d\n",
		posX, posY, posZ, val);
}
