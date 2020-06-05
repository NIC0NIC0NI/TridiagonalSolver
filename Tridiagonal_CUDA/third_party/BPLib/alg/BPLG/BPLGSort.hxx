//- =======================================================================
//+ BPLG Sort v1.0
//- =======================================================================

#pragma once
#ifndef BPLG_SORT_HXX
#define BPLG_SORT_HXX

//---- Header Dependencies -----------------------------------------------

#include "../Transform.hxx"
#include "tools/cuvector.hxx"

//---- Class Declaration -------------------------------------------------

class BPLGSort : public Transform {

public:

	// Creates a transform, batch is the last dimension
	BPLGSort(int dimX, int dimY = 1, int dimZ = 1) :
	  Transform(dimX, dimY, dimZ) { init(); }

	// Calculates the transform, direct (dir=1) or inverse (dir=-1)
	Transform* calc(int dir = 1);

	// Initializes data, sequencial (mode=0) or random (mode=1)
	inline Transform* init(int mode = -1) {
		init_(mode ? 1 : 1);
		updateGpu_();
		return this;
	}

	// Returns algorithm's name
	inline const char* toString() { return "BPLGSort"; }

	// Prints data
	Transform* print(const char* name = 0) {
		updateCpu_(); print_(name ? name : toString());
		return this; }




	// Compares relative error with respect to other algorithms
	inline double compare(Transform* ref, const char* label = 0) {
		
		updateCpu_(); //? if(ref) ref->updateCpu_c();
		compare_(ref, label); }

	// Returns mdata/sec instead of gflops in this case
	inline double gflops(double time, long long iters = 1) const {
		double nProb = (double)iters * (double)getZ();
		double data =  nProb * getX();
		if(time < 1e-3) time = 1e-3; // Preventing division by 0
		return data * 1e-6 / time;
	} 

};

#endif // BPLG_SORT_HXX
